#!/bin/bash
conda activate objnav
python -m sem_objnav.eval_objnav --checkpoint exp/rl_policy/0/model_3999622.zip --out_prefix 040_maskrcnn_hv --env_gpus 0 1 2 3 --num_workers 16 --val_scenes --exclude_classes plant --seg_prob_aggregate height_map_hv --seg_model_type maskrcnn --hv_ratio 0.3373 --hv_min_views 1 --hv_view_dist 4.9190
python -m sem_objnav.eval_objnav --checkpoint exp/rl_policy/0/model_3999622.zip --out_prefix 040_maskrcnn_goal_decay --env_gpus 0 1 2 3 --num_workers 16 --val_scenes --exclude_classes plant --seg_prob_aggregate height_map_goal_decay --seg_model_type maskrcnn --goal_decay 0.9155 --goal_mark 1.0743
python -m sem_objnav.eval_objnav --checkpoint exp/rl_policy/0/model_3999622.zip --out_prefix 040_maskrcnn_argmax --env_gpus 0 1 2 3 --num_workers 16 --val_scenes --exclude_classes plant --seg_prob_aggregate height_map_argmax --seg_model_type maskrcnn
