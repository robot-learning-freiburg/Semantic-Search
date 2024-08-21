import itertools
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class SemUncEvaluation:
    NUM_BINS = 10
    NUM_PROCESSES = 10
    BATCH_SIZE = 10

    def __init__(self, results_dir: Path, select_classes: str = None):
        self.results_dir = results_dir
        image_results_dir = results_dir / "val_results"
        self.gt_files, self.pred_files, self.probs_files = self.sort_files_by_extension(
            image_results_dir
        )
        self.select_classes = select_classes

    @staticmethod
    def sort_files_by_extension(directory):
        files = [
            list(directory.glob(f"*_{ext}.npy")) for ext in ["gt", "pred", "probs"]
        ]
        for file in files:
            file.sort()
        return files

    def evaluate_semantic_uncertainty(self):
        map_idxs = itertools.product(
            range(len(self.pred_files)), range(self.BATCH_SIZE)
        )
        pool = mp.Pool(processes=self.NUM_PROCESSES)
        results = pool.map(self.process_files, map_idxs)

        (
            overall_ece,
            bin_confidences,
            bin_accuracies,
        ) = self.calculate_ece_and_bins(results)

        print(f"Overall ECE: {overall_ece}")
        self.plot_results(bin_confidences, bin_accuracies, overall_ece)

        return overall_ece

    def process_files(self, batch_i):
        batch, i = batch_i
        sem_gt, sem_pred, probs = self.load_files(batch, i)
        sem_gt = sem_gt.astype(np.int32)
        sem_gt -= 1  # remove the void class
        if self.select_classes == "background":
            # ASSUMING THAT there are 13 background classes
            sem_gt[sem_gt >= 13] = -1
        elif self.select_classes == "search_objects":
            sem_gt[sem_gt < 13] = -1
        pred_prob = probs.max(axis=0)
        return self.calculate_bin_stats(sem_gt, sem_pred, pred_prob)

    def load_files(self, batch, i):
        files = [
            np.load(file[batch])[i]
            for file in [self.gt_files, self.pred_files, self.probs_files]
        ]
        return files

    def calculate_bin_stats(self, sem_gt, sem_pred, pred_prob):
        bin_accuracies, bin_confidences, bin_counts = (
            np.zeros(self.NUM_BINS),
            np.zeros(self.NUM_BINS),
            np.zeros(self.NUM_BINS),
        )
        valid_mask = sem_gt != -1
        for j in range(self.NUM_BINS):
            p1, p2 = j / self.NUM_BINS, (j + 1) / self.NUM_BINS
            confidence_bin_mask = (pred_prob > p1) & (pred_prob <= p2) & valid_mask

            bin_count = confidence_bin_mask.sum()
            if bin_count > 0:
                points_, labels_ = (
                    sem_pred[confidence_bin_mask],
                    sem_gt[confidence_bin_mask],
                )
                bin_accuracy = (points_ == labels_).sum()
                bin_confidence = pred_prob[confidence_bin_mask].sum()

                bin_accuracies[j] = bin_accuracy
                bin_confidences[j] = bin_confidence
                bin_counts[j] = bin_count
        return bin_accuracies, bin_confidences, bin_counts

    def calculate_ece_and_bins(self, results):
        bin_confidences, bin_accuracies, bin_counts = (
            np.zeros(self.NUM_BINS, dtype=np.float64),
            np.zeros(self.NUM_BINS, dtype=np.float64),
            np.zeros(self.NUM_BINS, dtype=np.float64),
        )

        for bin_accuracies_res, bin_confidences_res, bin_counts_res in results:
            bin_confidences += np.array(bin_confidences_res)
            bin_accuracies += np.array(bin_accuracies_res)
            bin_counts += np.array(bin_counts_res)
        bin_confidences = np.divide(
            bin_confidences,
            bin_counts,
            out=np.zeros_like(bin_confidences),
            where=bin_counts != 0,
        )
        bin_accuracies = np.divide(
            bin_accuracies,
            bin_counts,
            out=np.zeros_like(bin_accuracies),
            where=bin_counts != 0,
        )
        overall_ece = (
            np.sum(
                bin_counts * np.abs(bin_accuracies - bin_confidences),
            )
            / bin_counts.sum()
        )
        return (
            overall_ece,
            bin_confidences,
            bin_accuracies,
        )

    def plot_results(self, bin_confidences, bin_accuracies, overall_ece):
        plt.plot([0, 1], [0, 1], color="black", linewidth=2)
        plt.plot(bin_confidences, bin_accuracies, color="b")
        path = (
            self.results_dir
            / f"calibration_{self.select_classes or 'all'}_{overall_ece:.4f}.png"
        )
        plt.savefig(str(path), dpi=150)
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    for select_classes in ["background", "search_objects", None]:
        SemUncEvaluation(
            args.results_dir, select_classes
        ).evaluate_semantic_uncertainty()
