import json
import sys
from functools import partial

import optuna

from sem_objnav.eval_objnav import build_parser, main

_model_args = {
    "segformer": ["--seg_model_type", "segformer", "--temperature", "2.5"],
    "emsanet": [
        "--seg_model_type",
        "emsanet",
        "--seg_model",
        "../third_party/trained_models/sunrgbd/r34_NBt1D_pre.pth",
        "--seg_config",
        "../third_party/trained_models/sunrgbd/argsv.txt",
        "--temperature",
        "3.2",
    ],
    "maskrcnn": ["--seg_model_type", "maskrcnn"],
    "grounded_dino": [
        "--seg_model_type",
        "grounded_dino",
    ],
}


def hv_objective(trial: optuna.Trial, model: str = "segformer"):
    hv_ratio = trial.suggest_float("hv_ratio", 0.2, 1.0)
    hv_min_views = trial.suggest_int("hv_min_views", 1, 10)
    hv_view_dist = trial.suggest_float("hv_view_dist", 1.5, 5.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--hv_ratio",
            str(hv_ratio),
            "--hv_min_views",
            str(hv_min_views),
            "--hv_view_dist",
            str(hv_view_dist),
            "--seg_prob_aggregate",
            "height_map_hv",
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def argmax_objective(trial: optuna.Trial, model="grounded_dino"):
    assert model == "grounded_dino"
    seg_threshold = trial.suggest_float("seg_threshold", 0.2, 1.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "6",
            "--limit_scenes",
            "30",
            "--seg_threshold",
            str(seg_threshold),
            "--seg_prob_aggregate",
            "height_map_argmax",
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def hv_prob_objective(trial: optuna.Trial, model="segformer"):
    assert model != "maskrcnn"
    hv_ratio = trial.suggest_float("hv_ratio", 0.2, 1.0)
    hv_min_views = trial.suggest_int("hv_min_views", 1, 10)
    hv_view_dist = trial.suggest_float("hv_view_dist", 1.5, 5.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--hv_ratio",
            str(hv_ratio),
            "--hv_min_views",
            str(hv_min_views),
            "--hv_view_dist",
            str(hv_view_dist),
            "--seg_prob_aggregate",
            "height_map_prob_hv",
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def log_odds_objective(trial: optuna.Trial, model="segformer"):
    assert model != "maskrcnn"
    entropy_threshold = trial.suggest_float("entropy_threshold", 0.0, 1.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--entropy_threshold",
            str(entropy_threshold),
            "--seg_prob_aggregate",
            "height_map_log_odds",
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def filter_goal_prob_objective(trial: optuna.Trial, model="segformer"):
    assert model != "maskrcnn"
    filter_goal_prob = trial.suggest_float("filter_goal_prob", 0.2, 1.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--filter_goal_prob",
            str(filter_goal_prob),
            "--seg_prob_aggregate",
            "height_map_argmax",
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def goal_decay_objective(trial: optuna.Trial, model: str = "segformer"):
    goal_decay = trial.suggest_float("goal_decay", 0.5, 0.99)
    goal_mark = trial.suggest_float("goal_mark", 1.0, 3.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--goal_decay",
            str(goal_decay),
            "--goal_mark",
            str(goal_mark),
            "--seg_prob_aggregate",
            "height_map_goal_decay",
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def goal_decay_with_filter_objective(trial: optuna.Trial, model: str = "segformer"):
    assert model != "maskrcnn"

    goal_decay = trial.suggest_float("goal_decay", 0.5, 0.99)
    goal_mark = trial.suggest_float("goal_mark", 1.0, 3.0)
    filter_goal_prob = trial.suggest_float("filter_goal_prob", 0.2, 1.0)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--goal_decay",
            str(goal_decay),
            "--goal_mark",
            str(goal_mark),
            "--seg_prob_aggregate",
            "height_map_goal_decay",
            "--filter_goal_prob",
            str(filter_goal_prob),
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


def avg_prob_objective(
    trial: optuna.Trial,
    model: str = "segformer",
    agg_type="height_map_avg_prob",
    without_temp=False,
):
    assert model != "maskrcnn"

    entropy_threshold = trial.suggest_float("entropy_threshold", 0.0, 1.0)
    parser = build_parser()
    model_args = _model_args[model].copy()
    if without_temp:
        temp_idx = model_args.index("--temperature")
        model_args[temp_idx + 1] = "1.0"
    args = parser.parse_args(
        [
            "--exclude_classes",
            "plant",
            "--out_prefix",
            f"htune_{trial.number}",
            "--env_gpus",
            "0",
            "1",
            "2",
            "3",
            "--checkpoint",
            "exp/sp_policy",
            "--sp_policy",
            "--episodes",
            "1",
            "--num_workers",
            "10",
            "--limit_scenes",
            "30",
            "--entropy_threshold",
            str(entropy_threshold),
            "--seg_prob_aggregate",
            agg_type,
        ]
        + _model_args[model]
    )
    metrics = main(args)
    return -metrics["success"]


if __name__ == "__main__":
    obj_to_select = int(sys.argv[1])
    model = sys.argv[2]
    objective = [
        "hv_objective",
        "hv_prob_objective",
        "filter_goal_prob_objective",
        "goal_decay_objective",
        "goal_decay_with_filter_objective",
        "avg_prob_objective",
        "w_avg_prob_objective",
        "w2_avg_prob_objective",
        "log_odds_objective",
        "avg_prob_without_temp_objective",
        "w2_avg_prob_without_temp_objective",
        "argmax_objective",
    ][obj_to_select]
    objective_fn = {
        "hv_objective": hv_objective,
        "hv_prob_objective": hv_prob_objective,
        "filter_goal_prob_objective": filter_goal_prob_objective,
        "goal_decay_objective": goal_decay_objective,
        "avg_prob_objective": avg_prob_objective,
        "argmax_objective": argmax_objective,
        "goal_decay_with_filter_objective": goal_decay_with_filter_objective,
        "w_avg_prob_objective": partial(
            avg_prob_objective, agg_type="height_map_w_avg_prob"
        ),
        "w2_avg_prob_objective": partial(
            avg_prob_objective, agg_type="height_map_w2_avg_prob"
        ),
        "log_odds_objective": log_odds_objective,
        "avg_prob_without_temp_objective": partial(
            avg_prob_objective, without_temp=True
        ),
        "w2_avg_prob_without_temp_objective": partial(
            avg_prob_objective, agg_type="height_map_w2_avg_prob", without_temp=True
        ),
    }[objective]
    objective_fn = partial(objective_fn, model=model)
    study = optuna.create_study()
    study.optimize(objective_fn, n_trials=20)
    # save the best parameters in a json
    print(objective, study.best_params)
    with open(f"best_params_{objective}_{model}.json", "w") as f:
        json.dump(study.best_params, f)
