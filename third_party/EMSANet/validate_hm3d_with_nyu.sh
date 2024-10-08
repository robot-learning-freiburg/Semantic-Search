# Download https://drive.google.com/uc?id=1QbOJXVrOzsVM8ltX7AxqFSVLsY6vzvNX into trained_models
python -m sem_objnav.segment.hm3d.prepare_dataset --dataset_dir ../habitat-lab/data/scene_datasets/hm3d datasets/hm3d
python main.py --dataset hm3d --tasks semantic instance     --enable-panoptic     --rgb-encoder-backbone resnet34     --rgb-encoder-backbone-block nonbottleneck1d     --depth-encoder-backbone resnet34     --depth-encoder-backbone-block nonbottleneck1d     --no-pretrained-backbone     --input-modalities rgb depth --raw-depth  --weights-filepath ./trained_models/nyuv2/r34_NBt1D_pre.pth --validation-only --validation-split val --dataset-path ./datasets/hm3d --skip-sanity-check --n-workers 1

## RESULTS
# {'valid_instance_all_deeplab_num_categories': tensor(38),
#  'valid_instance_all_deeplab_pq': tensor(0.5803, dtype=torch.float64),
#  'valid_instance_all_deeplab_pq_best': tensor(0.5803, dtype=torch.float64),
#  'valid_instance_all_deeplab_rq': tensor(0.6686, dtype=torch.float64),
#  'valid_instance_all_deeplab_rq_best': tensor(0.6686, dtype=torch.float64),
#  'valid_instance_all_deeplab_sq': tensor(0.8361, dtype=torch.float64),
#  'valid_instance_all_deeplab_sq_best': tensor(0.8361, dtype=torch.float64),
#  'valid_instance_all_with_gt_deeplab_num_categories': tensor(38),
#  'valid_instance_all_with_gt_deeplab_pq': tensor(0.5803, dtype=torch.float64),
#  'valid_instance_all_with_gt_deeplab_pq_best': tensor(0.5803, dtype=torch.float64),
#  'valid_instance_all_with_gt_deeplab_rq': tensor(0.6686, dtype=torch.float64),
#  'valid_instance_all_with_gt_deeplab_rq_best': tensor(0.6686, dtype=torch.float64),
#  'valid_instance_all_with_gt_deeplab_sq': tensor(0.8361, dtype=torch.float64),
#  'valid_instance_all_with_gt_deeplab_sq_best': tensor(0.8361, dtype=torch.float64),
#  'valid_instance_center_loss_main': tensor([0.0034], device='cuda:0'),
#  'valid_instance_center_total_loss': tensor([0.0034], device='cuda:0'),
#  'valid_instance_epoch_end_time': 0.0018713580000166985,
#  'valid_instance_mae_deeplab_deg': tensor(nan, dtype=torch.float64),
#  'valid_instance_mae_deeplab_rad': tensor(nan, dtype=torch.float64),
#  'valid_instance_offset_loss_main': tensor([0.0361], device='cuda:0'),
#  'valid_instance_offset_total_loss': tensor([0.0361], device='cuda:0'),
#  'valid_instance_step_time': tensor([0.7883], device='cuda:0'),
#  'valid_instance_stuff_deeplab_num_categories': tensor(3),
#  'valid_instance_stuff_deeplab_pq': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_deeplab_pq_best': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_deeplab_rq': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_deeplab_rq_best': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_deeplab_sq': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_deeplab_sq_best': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_with_gt_deeplab_num_categories': tensor(3),
#  'valid_instance_stuff_with_gt_deeplab_pq': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_with_gt_deeplab_pq_best': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_with_gt_deeplab_rq': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_with_gt_deeplab_rq_best': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_with_gt_deeplab_sq': tensor(1., dtype=torch.float64),
#  'valid_instance_stuff_with_gt_deeplab_sq_best': tensor(1., dtype=torch.float64),
#  'valid_instance_things_deeplab_num_categories': tensor(35),
#  'valid_instance_things_deeplab_pq': tensor(0.5443, dtype=torch.float64),
#  'valid_instance_things_deeplab_pq_best': tensor(0.5443, dtype=torch.float64),
#  'valid_instance_things_deeplab_rq': tensor(0.6402, dtype=torch.float64),
#  'valid_instance_things_deeplab_rq_best': tensor(0.6402, dtype=torch.float64),
#  'valid_instance_things_deeplab_sq': tensor(0.8221, dtype=torch.float64),
#  'valid_instance_things_deeplab_sq_best': tensor(0.8221, dtype=torch.float64),
#  'valid_instance_things_with_gt_deeplab_num_categories': tensor(35),
#  'valid_instance_things_with_gt_deeplab_pq': tensor(0.5443, dtype=torch.float64),
#  'valid_instance_things_with_gt_deeplab_pq_best': tensor(0.5443, dtype=torch.float64),
#  'valid_instance_things_with_gt_deeplab_rq': tensor(0.6402, dtype=torch.float64),
#  'valid_instance_things_with_gt_deeplab_rq_best': tensor(0.6402, dtype=torch.float64),
#  'valid_instance_things_with_gt_deeplab_sq': tensor(0.8221, dtype=torch.float64),
#  'valid_instance_things_with_gt_deeplab_sq_best': tensor(0.8221, dtype=torch.float64),
#  'valid_panoptic_all_deeplab_num_categories': tensor(40),
#  'valid_panoptic_all_deeplab_pq': tensor(0.1911, dtype=torch.float64),
#  'valid_panoptic_all_deeplab_pq_best': tensor(0.1911, dtype=torch.float64),
#  'valid_panoptic_all_deeplab_rq': tensor(0.2369, dtype=torch.float64),
#  'valid_panoptic_all_deeplab_rq_best': tensor(0.2369, dtype=torch.float64),
#  'valid_panoptic_all_deeplab_sq': tensor(0.6369, dtype=torch.float64),
#  'valid_panoptic_all_deeplab_sq_best': tensor(0.6369, dtype=torch.float64),
#  'valid_panoptic_all_with_gt_deeplab_num_categories': tensor(38),
#  'valid_panoptic_all_with_gt_deeplab_pq': tensor(0.2012, dtype=torch.float64),
#  'valid_panoptic_all_with_gt_deeplab_pq_best': tensor(0.2012, dtype=torch.float64),
#  'valid_panoptic_all_with_gt_deeplab_rq': tensor(0.2494, dtype=torch.float64),
#  'valid_panoptic_all_with_gt_deeplab_rq_best': tensor(0.2494, dtype=torch.float64),
#  'valid_panoptic_all_with_gt_deeplab_sq': tensor(0.6704, dtype=torch.float64),
#  'valid_panoptic_all_with_gt_deeplab_sq_best': tensor(0.6704, dtype=torch.float64),
#  'valid_panoptic_deeplab_semantic_miou': tensor(0.2243),
#  'valid_panoptic_deeplab_semantic_miou_best': tensor(0.2243),
#  'valid_panoptic_epoch_end_time': 0.0006564619999949173,
#  'valid_panoptic_mae_deeplab_deg': tensor(nan, dtype=torch.float64),
#  'valid_panoptic_mae_deeplab_rad': tensor(nan, dtype=torch.float64),
#  'valid_panoptic_step_time': tensor([0.5462], device='cuda:0'),
#  'valid_panoptic_stuff_deeplab_num_categories': tensor(3),
#  'valid_panoptic_stuff_deeplab_pq': tensor(0.5466, dtype=torch.float64),
#  'valid_panoptic_stuff_deeplab_pq_best': tensor(0.5466, dtype=torch.float64),
#  'valid_panoptic_stuff_deeplab_rq': tensor(0.6616, dtype=torch.float64),
#  'valid_panoptic_stuff_deeplab_rq_best': tensor(0.6616, dtype=torch.float64),
#  'valid_panoptic_stuff_deeplab_sq': tensor(0.8247, dtype=torch.float64),
#  'valid_panoptic_stuff_deeplab_sq_best': tensor(0.8247, dtype=torch.float64),
#  'valid_panoptic_stuff_with_gt_deeplab_num_categories': tensor(3),
#  'valid_panoptic_stuff_with_gt_deeplab_pq': tensor(0.5466, dtype=torch.float64),
#  'valid_panoptic_stuff_with_gt_deeplab_pq_best': tensor(0.5466, dtype=torch.float64),
#  'valid_panoptic_stuff_with_gt_deeplab_rq': tensor(0.6616, dtype=torch.float64),
#  'valid_panoptic_stuff_with_gt_deeplab_rq_best': tensor(0.6616, dtype=torch.float64),
#  'valid_panoptic_stuff_with_gt_deeplab_sq': tensor(0.8247, dtype=torch.float64),
#  'valid_panoptic_stuff_with_gt_deeplab_sq_best': tensor(0.8247, dtype=torch.float64),
#  'valid_panoptic_things_deeplab_num_categories': tensor(37),
#  'valid_panoptic_things_deeplab_pq': tensor(0.1623, dtype=torch.float64),
#  'valid_panoptic_things_deeplab_pq_best': tensor(0.1623, dtype=torch.float64),
#  'valid_panoptic_things_deeplab_rq': tensor(0.2025, dtype=torch.float64),
#  'valid_panoptic_things_deeplab_rq_best': tensor(0.2025, dtype=torch.float64),
#  'valid_panoptic_things_deeplab_sq': tensor(0.6217, dtype=torch.float64),
#  'valid_panoptic_things_deeplab_sq_best': tensor(0.6217, dtype=torch.float64),
#  'valid_panoptic_things_with_gt_deeplab_num_categories': tensor(35),
#  'valid_panoptic_things_with_gt_deeplab_pq': tensor(0.1715, dtype=torch.float64),
#  'valid_panoptic_things_with_gt_deeplab_pq_best': tensor(0.1715, dtype=torch.float64),
#  'valid_panoptic_things_with_gt_deeplab_rq': tensor(0.2141, dtype=torch.float64),
#  'valid_panoptic_things_with_gt_deeplab_rq_best': tensor(0.2141, dtype=torch.float64),
#  'valid_panoptic_things_with_gt_deeplab_sq': tensor(0.6572, dtype=torch.float64),
#  'valid_panoptic_things_with_gt_deeplab_sq_best': tensor(0.6572, dtype=torch.float64),
#  'valid_semantic_epoch_end_time': 0.008937688000003163,
#  'valid_semantic_loss_main': tensor([1.6306], device='cuda:0'),
#  'valid_semantic_miou': tensor(0.2220),
#  'valid_semantic_miou_best': tensor(0.2220),
#  'valid_semantic_step_time': tensor([0.0837], device='cuda:0'),
#  'valid_semantic_total_loss': tensor([1.6306], device='cuda:0'),
#  'valid_total_loss': tensor([1.6736], device='cuda:0')}