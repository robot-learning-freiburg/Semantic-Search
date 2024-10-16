from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.weights import load_weights
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from transformers import (
    AutoModelForMaskGeneration,
    AutoProcessor,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    pipeline,
)

logger = logging.getLogger(__name__)


class EmsanetWrapper:
    def __init__(
        self,
        ckpt_path: str,
        config_path: str,
        map_schema: str,
        mode: str = "semantic",
        device: str = "cuda:0",
        temperature: float = 1.0,
    ) -> None:
        with open(config_path) as f:
            args = f.read().split()[1:]
        parser = ArgParserEMSANet()
        args = parser.parse_args(args)
        data = get_datahelper(args)
        model = EMSANet(args, data.dataset_config)
        self.num_model_classes = args.hm3d_semantic_n_classes
        with open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu")
            load_weights(args, model, ckpt["state_dict"], verbose=True)
        model = model.eval()
        self.model = model.cuda(device)
        self.preprocessor = get_preprocessor(
            args,
            dataset=data.datasets_valid[0],
            phase="test",
            multiscale_downscales=None,
        )
        self.device = device
        self.map_schema = map_schema
        self.mode = mode
        self.temp = temperature
        self.pred2objnav_occ_id = {
            1: 1,  # floor -> unoccupied
            4: 3,  # chair -> chair
            3: 4,  # bed -> bed
            32: 6,  # toilet -> toilet
            24: 7,  # monitor -> tv_monitor
            5: 8,  # sofa -> sofa
        }
        self.objnav2pred = defaultdict(list)
        for k, v in self.pred2objnav_occ_id.items():
            self.objnav2pred[v].append(k)

        assert mode in [
            "semantic",
            "panoptic",
        ], "mode must be 'semantic' or 'panoptic' for emsanet"

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        obs = {
            "depth": depth,
            "rgb": rgb,
        }
        processed_obs = self.preprocessor(obs)
        processed_obs = {
            k: v.unsqueeze(0).cuda(self.device) for k, v in processed_obs.items()
        }
        with torch.no_grad():
            output = self.model(processed_obs, do_postprocessing=True)
            logits = output["semantic_output"].squeeze() / self.temp
            probability = torch.nn.functional.softmax(logits, dim=0)
            uncertainty = -torch.sum(
                probability * torch.log(probability + 1e-6), dim=0
            ) / np.log(probability.shape[0])

            max_probs, semantic_seg = probability.max(dim=0)
            semantic_seg = self.map_seg_to_schema(semantic_seg.cpu().numpy()).astype(
                np.int32
            )
            max_probs = max_probs.cpu().numpy()
            probs = probability.cpu().numpy()
            if self.map_schema == "objnav_occ_id":
                for goal_obj_nav_seg_id in range(3, 9):
                    goal_pred_seg_ids = self.objnav2pred[goal_obj_nav_seg_id]
                    goal_mask = semantic_seg == goal_obj_nav_seg_id
                    # add the probabilities of all the prediction classes that map to the goal class
                    # to the goal class probability, we do this only for pixels that get
                    # argmax to one of the goal classes
                    max_probs[goal_mask] = probs[goal_pred_seg_ids, :].sum(axis=0)[
                        goal_mask
                    ]
            if self.mode == "panoptic":
                semantic_seg = (
                    output["panoptic_segmentation_deeplab_semantic_idx_fullres"] - 1
                )
                semantic_seg = semantic_seg.long().squeeze().cpu().numpy()
                semantic_seg = self.map_seg_to_schema(semantic_seg)
        return {
            "semantic": semantic_seg,
            "semantic_max_prob": max_probs,
            "semantic_prob": probability.cpu().numpy(),
            "semantic_unc": uncertainty.cpu().numpy(),
            "logits": output["semantic_output"].cpu().numpy(),
        }

    def map_seg_to_schema(self, seg):
        if self.map_schema == "objnav_occ_id" and self.num_model_classes == 37:
            seg_pred = np.ones_like(seg) * 2
            for key, value in self.pred2objnav_occ_id.items():
                seg_pred[seg == key] = value
        return seg_pred


class MaskRCNNWrapper:
    def __init__(
        self,
        map_schema: str,
        device: str = "cuda:0",
    ) -> None:
        self.map_schema = map_schema
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        model = model.eval()
        self.model = model.cuda(device)
        self.map_schema = map_schema
        # ["void", "chair", "bed", "plant", "toilet", "tv", "sofa"]
        self.goal_classes = [62, 65, 64, 70, 72, 63]
        self.device = device
        self.num_model_classes = None
        self.objnav2pred = None

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        image_tensor = torchvision.transforms.functional.to_tensor(rgb)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            predictions = self.model([image_tensor])

        pred_class_ids = predictions[0]["labels"].cpu().numpy()
        pred_masks = predictions[0]["masks"]
        pred_boxes = predictions[0]["boxes"]
        pred_scores = predictions[0]["scores"]

        pred_seg = np.zeros_like(depth)
        for pred_class_id, pred_mask, pred_score in zip(
            pred_class_ids, pred_masks, pred_scores
        ):
            if pred_score < 0.5:
                break
            if pred_class_id not in self.goal_classes:
                continue
            seg_id = self.goal_classes.index(pred_class_id)
            if self.map_schema == "objnav_occ_id":
                seg_id += 3
            else:
                raise NotImplementedError
            pred_mask = pred_mask.squeeze().cpu().numpy()
            pred_mask[pred_mask > 0.5] = 1
            pred_mask[pred_mask <= 0.5] = 0
            pred_seg[((pred_mask == 1) & (pred_seg == 0))] = seg_id
        return {"semantic": pred_seg.astype(np.int32)}

    def map_seg_to_schema(self, seg):
        raise NotImplementedError


class OneformerWrapper:
    def __init__(
        self,
        map_schema: str,
        model: str = "shi-labs/oneformer_ade20k_swin_large",
        device: str = "cuda:0",
        mode: str = "semantic",
    ) -> None:
        self.processor = OneFormerProcessor.from_pretrained(model)
        model = OneFormerForUniversalSegmentation.from_pretrained(model)
        self.model = model.to(device).eval()

        self.map_schema = map_schema
        self.mode = mode
        assert mode in [
            "semantic",
            "panoptic",
        ], "mode must be 'semantic' or 'panoptic' for oneformer"
        self.device = device
        self.num_model_classes = 150
        assert map_schema == "objnav_occ_id", "map_schema must be 'objnav_occ_id'"
        # ["void", "chair", "bed", "plant", "toilet", "tv", "sofa"]
        self.goal_classes = [62, 65, 64, 70, 72, 63]
        self.pred2objnav_occ_id = {
            3: 1,  # floor -> unoccupied
            0: 2,  # wall -> occupied
            19: 3,  # chair -> chair
            30: 3,  # armchair -> chair
            75: 3,  # swivelchair -> chair
            7: 4,  # bed -> bed
            17: 5,  # plant -> plant
            65: 6,  # toilet -> toilet
            143: 7,  # monitor -> tv_monitor
            130: 7,  # screen -> tv_monitor
            141: 7,  # crt screen -> tv_monitor
            89: 7,  # television receiver -> tv_monitor
            23: 8,  # sofa -> sofa
        }

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        rgb = rgb.transpose(2, 0, 1)
        inputs = self.processor(
            images=rgb, task_inputs=[self.mode], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_out = self.model(**inputs)
            if self.mode == "semantic":
                pred_seg = self.processor.post_process_semantic_segmentation(
                    model_out, target_sizes=[rgb.shape[1:]]
                )[0].cpu()
            else:
                panoptic_outs = self.processor.post_process_panoptic_segmentation(
                    model_out, target_sizes=[rgb.shape[1:]]
                )[0]
                pred_seg = panoptic_outs["segmentation"]
                for item in panoptic_outs["segments_info"]:
                    pred_seg[pred_seg == item["id"]] = item["label_id"]
                pred_seg = pred_seg.cpu()
        return {"semantic": self.map_seg_to_schema(pred_seg).astype(np.int32)}

    def map_seg_to_schema(self, seg):
        if self.map_schema == "objnav_occ_id":
            seg_pred = np.ones_like(seg) * 2
            for key, value in self.pred2objnav_occ_id.items():
                seg_pred[seg == key] = value
        return seg_pred


class SegformerWrapper:
    def __init__(
        self,
        map_schema: str,
        model: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str = "cuda:0",
        temperature: float = 1.0,
    ) -> None:
        self.processor = SegformerFeatureExtractor.from_pretrained(model)
        model = SegformerForSemanticSegmentation.from_pretrained(model)
        self.model = model.to(device).eval()
        self.map_schema = map_schema
        self.temp = temperature
        self.device = device
        self.num_model_classes = 150
        assert map_schema == "objnav_occ_id", "map_schema must be 'objnav_occ_id'"
        # ["void", "chair", "bed", "plant", "toilet", "tv", "sofa"]
        self.goal_classes = [62, 65, 64, 70, 72, 63]
        self.pred2objnav_occ_id = {
            3: 1,  # floor -> unoccupied
            0: 2,  # wall -> occupied
            19: 3,  # chair -> chair
            30: 3,  # armchair -> chair
            75: 3,  # swivelchair -> chair
            7: 4,  # bed -> bed
            17: 5,  # plant -> plant
            65: 6,  # toilet -> toilet
            143: 7,  # monitor -> tv_monitor
            130: 7,  # screen -> tv_monitor
            141: 7,  # crt screen -> tv_monitor
            89: 7,  # television receiver -> tv_monitor
            23: 8,  # sofa -> sofa
        }
        self.objnav2pred = defaultdict(list)
        for k, v in self.pred2objnav_occ_id.items():
            self.objnav2pred[v].append(k)

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        rgb = rgb.transpose(2, 0, 1)
        inputs = self.processor(rgb, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = (
                nn.functional.interpolate(
                    outputs.logits,
                    size=rgb.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
            )[0] / self.temp
            probs = torch.nn.functional.softmax(logits, dim=0)
            probs_seg, pred_seg = probs.max(dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=0) / np.log(
                probs.shape[0]
            )
            pred_seg = self.map_seg_to_schema(pred_seg.cpu().numpy()).astype(np.int32)
            probs_seg = probs_seg.cpu().numpy()
            probs = probs.cpu().numpy()
            if self.map_schema == "objnav_occ_id":
                for goal_obj_nav_seg_id in range(3, 9):
                    goal_pred_seg_ids = self.objnav2pred[goal_obj_nav_seg_id]
                    goal_mask = pred_seg == goal_obj_nav_seg_id
                    # add the probabilities of all the prediction classes that map to the goal class
                    # to the goal class probability, we do this only for pixels that get
                    # argmax to one of the goal classes
                    probs_seg[goal_mask] = probs[goal_pred_seg_ids, :].sum(axis=0)[
                        goal_mask
                    ]
        return {
            "semantic": pred_seg,
            "semantic_max_prob": probs_seg,
            "semantic_prob": probs,
            "semantic_unc": entropy.cpu().numpy(),
            "logits": logits.cpu().numpy(),
        }

    def map_seg_to_schema(self, seg):
        if self.map_schema == "objnav_occ_id":
            seg_pred = np.ones_like(seg) * 2
            for key, value in self.pred2objnav_occ_id.items():
                seg_pred[seg == key] = value
        return seg_pred


class GroundedDinoWrapper:
    """
    GroundedDinoWrapper is a wrapper for the GroundedDino model.
    Inspired from: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
    """

    def __init__(
        self,
        map_schema: str,
        device: str = "cuda:0",
    ) -> None:
        self.map_schema = map_schema
        self.labels = [
            # "a floor.",
            # "a wall.",
            "a chair.",
            "a bed.",
            "a plant.",
            "a toilet.",
            "a TV.",
            "a sofa.",
        ]
        assert map_schema == "objnav_occ_id", "map_schema must be 'objnav_occ_id'"
        self.object_detector = pipeline(
            model="IDEA-Research/grounding-dino-tiny",
            task="zero-shot-object-detection",
            device=device,
        )
        segmentor_id = "facebook/sam-vit-base"
        self.segmentator = (
            AutoModelForMaskGeneration.from_pretrained(segmentor_id).to(device).eval()
        )
        self.processor = AutoProcessor.from_pretrained(segmentor_id)
        self.device = device
        self.num_model_classes = None
        self.objnav2pred = None
        self.threshold = 0.9

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        # convert rgb to PIL image
        rgb = Image.fromarray(rgb)
        pred_seg = np.zeros_like(depth)
        with torch.no_grad():
            results = self.object_detector(
                rgb, candidate_labels=self.labels, threshold=self.threshold
            )
            detection_results = [
                DetectionResult.from_dict(result) for result in results
            ]
            boxes = get_boxes(detection_results)
            if len(boxes[0]) > 0:
                inputs = self.processor(
                    images=rgb, input_boxes=boxes, return_tensors="pt"
                ).to(self.device)
                outputs = self.segmentator(**inputs)
                masks = self.processor.post_process_masks(
                    masks=outputs.pred_masks,
                    original_sizes=inputs.original_sizes,
                    reshaped_input_sizes=inputs.reshaped_input_sizes,
                )[0]
                masks = refine_masks(masks, polygon_refinement=False)
                for detection_result, mask in zip(detection_results, masks):
                    label_idx = self.labels.index(detection_result.label)
                    print(detection_result.label)
                    # if label_idx < 2:
                    #     continue
                    if self.map_schema == "objnav_occ_id":
                        label_idx += 3
                    pred_seg[mask > 0] = label_idx

        return {"semantic": pred_seg.astype(np.int32)}

    def map_seg_to_schema(self, seg):
        raise NotImplementedError


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask
    return masks
