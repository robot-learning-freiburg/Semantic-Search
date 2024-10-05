# Perception Matters: Enhancing Embodied AI with Uncertainty-Aware Semantic Segmentation
[**arXiv**](https://arxiv.org/abs/2408.02297) | [**website**](http://semantic-search.cs.uni-freiburg.de/) 

Repository providing the source code for the paper
>Perception Matters: Enhancing Embodied AI with Uncertainty-Aware Semantic Segmentation
>[Sai Prasanna](https://saiprasanna.in/about), [Daniel Honerkamp](https://rl.uni-freiburg.de/people/honerkamp)* [Kshitij Sirohi](http://www2.informatik.uni-freiburg.de/~sirohik/)*,  [Tim Welschehold](https://rl.uni-freiburg.de/people/welschehold), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard-2/) and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)

<!--
<p align="center">
  <img src="assets/overview.png" alt="Overview" width="1200" />
</p>
-->

Please cite the paper as follows:

    @article{prasanna2024perception,
      title={Perception Matters: Enhancing Embodied AI with Uncertainty-Aware Semantic Segmentation},
      author={Sai Prasanna and Daniel Honerkamp and Kshitij Sirohi and Tim Welschehold and Wolfram Burgard and Abhinav Valada},
      journal={Proceedings of the International Symposium on Robotics Research (ISRR)},
      year={2024}
    }


## Setup

1. You have to obtain the API user name and token for hm3d dataset from matterport by following their [instructions](https://matterport.com/partners/meta). Set these as environment variables `export USERNAME=<API_TOKEN_USER_ID>` `export PASSWORD=<API_TOKEN>`.
2. Run the setup.sh to create the conda environment.
3. Download the EMSANet checkpoint from `https://drive.google.com/uc?id=1LD4_g-jL4KJPRUmCGgXxx2xGQ7TNZ_o2` and extract it `tar -xvf checkpoint.tar.gz -C ./third_party/trained_models/`

## Evaluating aggregation approaches with the Shortest path policy

To evaluate the aggregation approaches with the shortest path policy, run 

```
./scripts/eval_sp_policy_emsanet.sh
./scripts/eval_sp_policy_maskrcnn.sh
./scripts/eval_sp_policy_segformer.sh
```

## Training and evaluating RL Policy

To train the RL policy on ground truth semantics and evaluate it with different semantic models and aggregation approaches, run

```
./scripts/train_rl_policy.sh
./scripts/eval_rl_policy_emsanet.sh
./scripts/eval_rl_policy_maskrcnn.sh
./scripts/eval_rl_policy_segformer.sh
```

# Misc

## Calibrating the perception model

1. Collect the data for calibrating the perception model. Run 
```
python -m sem_objnav.obj_nav.collect_seg_data --output_dir calibation_dataset
```
2. Check the notebooks `sem_objnav/notebooks/emsanet_scaling_temp.ipynb` and `sem_objnav/notebooks/segformer_scaling_temp.ipynb` for calibation.

## Stubborn

To collect data and train the models used in stubborn, run `./scripts/train_stubborn.sh`.

## Hyperparameter optimization

To find optimal hyperparameters for the aggregation strategies, run `./scripts/htune.sh`.
