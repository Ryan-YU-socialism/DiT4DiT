import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path

logger = get_logger(__name__)

_DISCRETE_ACTION_HEAD_TYPES = {"discrete", "categorical", "discrete_action"}


def _config_values_equal(existing, expected) -> bool:
    if isinstance(expected, list):
        try:
            return list(existing) == expected
        except TypeError:
            return False
    return existing == expected


def synchronize_discrete_action_data_config(cfg, data_cfg) -> None:
    """Derive discrete dataset semantics from the action-head configuration.

    This keeps the data labels and checkpoint class order from silently
    drifting apart. Explicit dataset values remain supported, but a conflict is
    rejected instead of being overwritten.
    """
    action_cfg = cfg.framework.action_model
    action_head_type = str(action_cfg.get("action_head_type", "flowmatching")).lower()
    if action_head_type not in _DISCRETE_ACTION_HEAD_TYPES:
        return

    action_horizon = int(
        action_cfg.get(
            "action_horizon",
            int(action_cfg.get("future_action_window_size", 63)) + 1,
        )
    )
    action_dim = int(action_cfg.get("action_dim", 3))
    class_values = action_cfg.get("action_class_values", None)
    if class_values is None:
        raise ValueError(
            "A discrete action head requires explicit action_class_values."
        )
    class_values = list(class_values)
    target_format = str(action_cfg.get("action_target_format", "values"))

    derived = {
        "discrete_actions": True,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "max_action_dim": action_dim,
        "action_delta_indices": list(range(action_horizon)),
        "action_class_values": class_values,
        "action_target_format": target_format,
    }
    conflicts = []
    for key, expected in derived.items():
        existing = data_cfg.get(key, None)
        if existing is not None and not _config_values_equal(existing, expected):
            conflicts.append(f"{key}: dataset={existing!r}, action_model={expected!r}")
    if conflicts:
        raise ValueError(
            "Discrete dataset configuration conflicts with framework.action_model: "
            + "; ".join(conflicts)
        )

    for key, value in derived.items():
        data_cfg[key] = value

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"):

    if dataset_py == "lerobot_datasets":
        from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        synchronize_discrete_action_data_config(cfg, vla_dataset_cfg)

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
        
    else:
        raise NotImplementedError(f"Dataset {dataset_py} is not supported yet")
        
