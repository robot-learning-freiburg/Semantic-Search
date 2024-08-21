#!/bin/bash
conda activate objnav
mkdir -p exp/sp_policy_stubborn_data
python -m sem_objnav.eval_objnav --checkpoint exp/sp_policy_stubborn_data --out_prefix segformer_train --env_gpus 0 1 2 3 --num_workers 16 --limit_scenes 64 --episodes 1 --exclude_classes plant --seg_prob_aggregate height_map_argmax --seg_model_type segformer --sp_policy --collect_nb_data
python -m sem_objnav.train_stubborn_nb_models --data exp/sp_policy_stubborn_data/segformer --save_path sem_objnav/obj_nav/nb_models/segformer.pkl
python -m sem_objnav.eval_objnav --checkpoint exp/sp_policy_stubborn_data --out_prefix emsanet --env_gpus 0 1 2 3 --num_workers 16 --limit_scenes 64 --episodes 1 --exclude_classes plant --seg_prob_aggregate height_map_argmax --seg_model_type emsanet --seg_model ../third_party/trained_models/sunrgbd/r34_NBt1D_pre.pth --seg_config ../third_party/trained_models/sunrgbd/argsv.txt --sp_policy --collect_nb_data 
python -m sem_objnav.train_stubborn_nb_models --data_path exp/sp_policy_stubborn_data/emsanet --save_path sem_objnav/obj_nav/nb_models/emsanet.pkl
