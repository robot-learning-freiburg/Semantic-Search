#!/bin/bash
conda activate objnav
for task_id in 0 1 2 3 4 5 6 7 8 9 10; do
    python -m sem_objnav.htune_evalobjnav $task_id segformer
    python -m sem_objnav.htune_evalobjnav $task_id emsanet
done
for task_id in 0 3; do
    python -m sem_objnav.htune_evalobjnav $task_id maskrcnn
done
mkdir best_hparams
mv best_params*.json best_hparams/