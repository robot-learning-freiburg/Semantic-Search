# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize_predictions
from nicr_mt_scene_analysis.data import move_batch_to_device, mt_collate
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig, DepthStats


def _get_args():
    parser = ArgParserEMSANet()

    # add additional arguments
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
        "they are most likely not valid. Note, this clipping is applied "
        "before scaling the depth values.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply.",
    )

    args = parser.parse_args()

    # this makes sure that visualization works
    args.visualize_validation = True

    return args


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    args = _get_args()
    assert all(
        x in args.input_modalities for x in ("rgb", "depth")
    ), "Only RGBD inference supported so far"

    device = torch.device("cuda")

    # data and model
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    dataset_config = DatasetConfig(
        dataset_config.semantic_label_list,
        dataset_config.semantic_label_list_without_void,
        dataset_config.scene_label_list,
        dataset_config.scene_label_list_without_void,
        Hm3dMeta.TRAIN_SPLIT_DEPTH_STATS,
    )

    model = EMSANet(args, dataset_config=dataset_config)

    # load weights
    checkpoint = torch.load(args.weights_filepath)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    print(f"Loading checkpoint: '{args.weights_filepath}'")

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # build preprocessor
    preprocessor = get_preprocessor(
        args, dataset=data.datasets_valid[0], phase="test", multiscale_downscales=None
    )

    # get samples
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    rgb_filepaths = sorted(glob(os.path.join(basepath, "*_rgb*.npy")))
    depth_filepaths = sorted(glob(os.path.join(basepath, "*_depth*.npy")))
    ground_truth_filepaths = sorted(glob(os.path.join(basepath, "*_truth*.npy")))
    assert len(rgb_filepaths) == len(depth_filepaths)
    for fp_rgb, fp_depth, gt_semantic in zip(
        rgb_filepaths, depth_filepaths, ground_truth_filepaths
    ):
        # load rgb and depth image
        img_rgb = np.load(fp_rgb)  # _load_img(fp_rgb)
        img_depth = np.load(fp_depth)  # _load_img(fp_depth).astype('float32')
        if args.depth_max is not None:
            img_depth[img_depth > args.depth_max] = 0
        img_depth *= args.depth_scale

        # preprocess sample
        sample = preprocessor(
            {
                "rgb": img_rgb,
                "depth": img_depth,
                "identifier": os.path.basename(os.path.splitext(fp_rgb)[0]),
            }
        )

        # add batch axis as there is no dataloader
        batch = mt_collate([sample])
        batch = move_batch_to_device(batch, device=device)

        # apply model
        prediction = model(batch, do_postprocessing=True)

        # visualize predictions
        prediction_visualization = visualize_predictions(
            prediction, batch, dataset_config
        )

        # show results
        fig, axs = plt.subplots(2, 5, figsize=(12, 6), dpi=150)
        [ax.set_axis_off() for ax in axs.ravel()]
        axs[0, 0].set_title("rgb")
        axs[0, 0].imshow(img_rgb)
        axs[0, 1].set_title("depth")
        axs[0, 1].imshow(img_depth)
        axs[0, 2].set_title("semantic")
        axs[0, 2].imshow(prediction_visualization["semantic"][0])
        axs[0, 3].set_title("panoptic")
        axs[0, 3].imshow(prediction_visualization["panoptic"][0])
        axs[1, 0].set_title("instance")
        axs[1, 0].imshow(prediction_visualization["instance"][0])
        axs[1, 1].set_title("instance center")
        axs[1, 1].imshow(prediction_visualization["instance_center"][0])
        axs[1, 2].set_title("instance offset")
        axs[1, 2].imshow(prediction_visualization["instance_offset"][0])
        axs[1, 3].set_title("panoptic with orientation")
        axs[1, 3].imshow(prediction_visualization["panoptic_orientation"][0])
        axs[0, 4].set_title("panoptic")
        axs[0, 4].imshow(prediction["panoptic_segmentation_deeplab"][0])
        axs[1, 4].set_title("semantic ground truth")
        dummy = {
            "semantic_segmentation_idx_fullres": torch.tensor(
                np.load(gt_semantic)
            ).unsqueeze(0)
        }
        axs[1, 4].imshow(
            visualize_predictions(dummy, batch, dataset_config)["semantic"][0]
        )

        plt.suptitle(
            f"Image: ({os.path.basename(fp_rgb)}, "
            f"{os.path.basename(fp_depth)}), "
            f"Model: {args.weights_filepath}"
        )
        plt.tight_layout()

        # fp = os.path.join('./', 'samples', f'result_{args.dataset}.png')
        # plt.savefig(fp, bbox_inches='tight', pad_inches=0.05, dpi=150)

        plt.show(block=True)


if __name__ == "__main__":
    main()
