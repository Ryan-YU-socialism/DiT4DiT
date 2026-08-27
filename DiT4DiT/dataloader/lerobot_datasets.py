# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from DiT4DiT.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from DiT4DiT.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from DiT4DiT.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from DiT4DiT.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from DiT4DiT.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from DiT4DiT.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor


def collate_fn(batch):
    return batch


def _configured_modality_config(data_cfg: dict) -> dict[str, ModalityConfig] | None:
    """Build modality sampling from explicit dataset keys when provided.

    No EndoMotion column names are assumed here. A dataset whose keys do not
    match an existing robot config can provide all four modality key lists in
    ``modality_keys``.
    """
    configured_keys = data_cfg.get("modality_keys", None)
    if configured_keys is None:
        return None

    required_modalities = ("video", "state", "action", "language")
    keys_by_modality: dict[str, list[str]] = {}
    for modality in required_modalities:
        keys = configured_keys.get(modality, None)
        if keys is None or len(keys) == 0:
            raise ValueError(
                "Configured datasets must provide non-empty modality_keys for "
                f"{modality!r}."
            )
        keys_by_modality[modality] = list(keys)

    observation_delta = list(data_cfg.get("observation_delta_indices", [0]))
    action_delta = data_cfg.get("action_delta_indices", None)
    if action_delta is None:
        action_delta = list(range(int(data_cfg.get("action_horizon", 64))))
    else:
        action_delta = list(action_delta)

    return {
        "video": ModalityConfig(
            delta_indices=observation_delta,
            modality_keys=keys_by_modality["video"],
        ),
        "state": ModalityConfig(
            delta_indices=observation_delta,
            modality_keys=keys_by_modality["state"],
        ),
        "action": ModalityConfig(
            delta_indices=action_delta,
            modality_keys=keys_by_modality["action"],
        ),
        "language": ModalityConfig(
            delta_indices=observation_delta,
            modality_keys=keys_by_modality["language"],
        ),
    }


def _without_action_transforms(
    transforms: ComposedModalityTransform,
    action_keys: list[str],
) -> ComposedModalityTransform:
    """Keep video/state transforms while preserving categorical action labels."""
    action_key_set = set(action_keys)
    retained = []
    for transform in transforms.transforms:
        # ConcatTransform consumes the original action keys, while the mixture
        # loader intentionally concatenates them after discrete validation.
        if getattr(transform, "action_concat_order", None) is not None:
            continue

        apply_to = set(getattr(transform, "apply_to", []))
        action_overlap = apply_to.intersection(action_key_set)
        if action_overlap:
            # Tensor conversion preserves {-1, 0, +1}, including when one
            # conversion transform covers both state and action keys.
            if isinstance(transform, StateActionToTensor):
                retained.append(transform)
                continue
            non_action_keys = apply_to.difference(action_key_set)
            if non_action_keys:
                raise ValueError(
                    "A transform mixes discrete action keys with other modalities; "
                    "split that transform before enabling discrete_actions."
                )
            # Normalization, rotation conversion, perturbation, and action
            # augmentation are omitted for already-categorical labels.
            continue
        retained.append(transform)
    return ComposedModalityTransform(transforms=retained)


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    discrete_actions = bool(data_cfg.get("discrete_actions", False)) if data_cfg else False
    configured_modalities = (
        _configured_modality_config(data_cfg) if discrete_actions and data_cfg else None
    )
    if configured_modalities is not None:
        modality_config = configured_modalities
        # Tensor conversion is semantics-preserving. With unknown keys we do
        # not guess any normalization, rotation conversion, or augmentation.
        # Image conversion/resizing is handled downstream by the mixture loader.
        transforms = ComposedModalityTransform(
            transforms=[
                StateActionToTensor(
                    apply_to=modality_config["state"].modality_keys,
                ),
                StateActionToTensor(
                    apply_to=modality_config["action"].modality_keys,
                ),
            ]
        )
    else:
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise ValueError(
                f"Unknown robot_type {robot_type!r}. Provide explicit modality_keys "
                "for an unregistered discrete dataset."
            )
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        modality_config = data_config.modality_config()
        transforms = data_config.transform()
        if discrete_actions:
            transforms = _without_action_transforms(
                transforms,
                modality_config["action"].modality_keys,
            )
    dataset_path = Path(data_root_dir) / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
    
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.get("data_root_dir", None)
    if data_root_dir is None:
        raise ValueError("data_root_dir must be configured.")
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    dataset_name = data_cfg.get("dataset_name", None)
    if dataset_name is not None:
        robot_type = data_cfg.get("robot_type", "configured_discrete")
        mixture_spec = [(str(dataset_name), 1.0, str(robot_type))]
    else:
        data_mix = data_cfg.get("data_mix", None)
        if data_mix is None:
            raise ValueError("Configure either dataset_name or data_mix.")
        mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
