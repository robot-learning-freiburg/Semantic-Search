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
from emsanet.visualization import visualize_batches, visualize_predictions
from emsanet.weights import load_weights
from nicr_mt_scene_analysis.data import move_batch_to_device, mt_collate
from sem_objnav.segment.hm3d.hm3d import Hm3dMeta


def _get_args():
    parser = ArgParserEMSANet()

    # add additional arguments
    group = parser.add_argument_group("Inference")
    group.add_argument(  # useful for appm context module
        "--inference-input-height",
        type=int,
        default=480,
        dest="validation_input_height",  # used in test phase
        help="Network input height for predicting on inference data.",
    )
    group.add_argument(  # useful for appm context module
        "--inference-input-width",
        type=int,
        default=640,
        dest="validation_input_width",  # used in test phase
        help="Network input width for predicting on inference data.",
    )
    group.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
        "they are most likely not valid. Note, this clipping is applied "
        "before scaling the depth values.",
    )
    group.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply.",
    )

    return parser.parse_args()


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
    model = EMSANet(args, dataset_config=dataset_config)

    # load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath)
    state_dict = checkpoint["state_dict"]
    if "epoch" in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    load_weights(args, model, state_dict, verbose=True)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # build preprocessor
    preprocessor = get_preprocessor(
        args, dataset=data.datasets_valid[0], phase="test", multiscale_downscales=None
    )

    # get samples
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    rgb_filepaths = sorted(glob(os.path.join(basepath, "*_rgb.*")))
    depth_filepaths = sorted(glob(os.path.join(basepath, "*_depth.*")))
    instance_filepaths = sorted(glob(os.path.join(basepath, "val/instance/*")))
    semantic_filepaths = sorted(glob(os.path.join(basepath, "val/semantic_81/*")))
    # parent of __file__
    dir_path = os.path.dirname(os.path.abspath(__file__))
    semantic_d2_pred = sorted(
        glob(
            os.path.join(
                dir_path,
                "external/output/inference/hm3d_val/panoptic_pred/semantic_pred/epoch_1/*.png",
            )
        )
    )
    assert len(rgb_filepaths) == len(depth_filepaths)
    sample_idx = np.random.RandomState(1337).choice(
        len(rgb_filepaths), size=30, replace=False
    )

    for idx in sample_idx:
        fp_rgb = rgb_filepaths[idx]
        fp_depth = depth_filepaths[idx]
        fp_instance = instance_filepaths[idx]
        fp_semantic = semantic_filepaths[idx]
        fp_d2_semantic = semantic_d2_pred[idx]
        # load rgb and depth image
        img_rgb = _load_img(fp_rgb)
        img_depth = _load_img(fp_depth).astype("float32")
        img_semantic = _load_img(fp_semantic)
        img_instance = _load_img(fp_instance)
        img_semantic_d2 = _load_img(fp_d2_semantic)
        img_semantic_d2 = np.array(Hm3dMeta.SEMANTIC_CLASS_COLORS_81[1:])[
            img_semantic_d2
        ]
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
        predictions = model(batch, do_postprocessing=True)

        # visualize predictions
        batch_visualization = visualize_batches(batch, dataset_config)
        preds_viz = visualize_predictions(
            predictions=predictions, batch=batch, dataset_config=dataset_config
        )
        visualize_sample(batch_visualization, preds_viz, img_semantic_d2, 0)
        fp = os.path.join("./EMSANet/", "samples", f"result_{args.dataset}_{idx}.png")


def visualize_sample(
    batch_visualization, prediction_visualization, img_semantic_d2, idx
):
    fig, axs = plt.subplots(5, 2, figsize=(6, 12))
    [ax.set_axis_off() for ax in axs.ravel()]
    axs[0, 0].set_title("rgb_input")
    axs[0, 0].imshow(batch_visualization["rgb"][idx])
    axs[0, 1].set_title("depth_input")
    axs[0, 1].imshow(batch_visualization["depth"][idx])
    axs[1, 0].set_title("sematic_gt")
    axs[1, 0].imshow(batch_visualization["semantic"][idx])
    axs[1, 1].set_title("instance_gt")
    axs[1, 1].imshow(batch_visualization["instance"][idx])
    axs[2, 0].set_title("semantic_pred")
    axs[2, 0].imshow(prediction_visualization["semantic"][idx])
    axs[2, 1].set_title("instance_pred")
    axs[2, 1].imshow(prediction_visualization["instance"][idx])
    axs[3, 0].set_title("panoptic_pred")
    axs[3, 0].imshow(prediction_visualization["panoptic"][idx])
    axs[3, 1].set_title("instance_offset_pred")
    axs[3, 1].imshow(prediction_visualization["instance_offset"][idx])
    axs[4, 0].set_title("semantic_pred_detectron2")
    axs[4, 0].imshow(img_semantic_d2)


if __name__ == "__main__":
    main()
