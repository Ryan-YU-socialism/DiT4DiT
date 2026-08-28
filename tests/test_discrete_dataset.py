import numpy as np
import pytest
from types import SimpleNamespace

from DiT4DiT.dataloader import synchronize_discrete_action_data_config
from DiT4DiT.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from DiT4DiT.dataloader.gr00t_lerobot.datasets import (
    build_action_valid_mask,
    get_shared_action_delta_indices,
    validate_discrete_action_mapping,
    validate_discrete_action_target,
)
from DiT4DiT.dataloader.gr00t_lerobot.embodiment_tags import (
    EmbodimentTag,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
)
from DiT4DiT.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from DiT4DiT.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from DiT4DiT.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from DiT4DiT.dataloader.lerobot_datasets import (
    _configured_modality_config,
    _without_action_transforms,
)


def test_action_valid_mask_marks_episode_padding_per_axis():
    mask = build_action_valid_mask(
        [0, 1, 2, 3],
        base_index=2,
        trajectory_length=5,
        action_dim=3,
    )

    assert mask.shape == (4, 3)
    np.testing.assert_array_equal(mask[:, 0], [True, True, True, False])
    np.testing.assert_array_equal(mask[:, 0], mask[:, 1])
    np.testing.assert_array_equal(mask[:, 1], mask[:, 2])


def test_values_format_ignores_only_masked_padding():
    action = np.zeros((64, 3), dtype=np.float32)
    mask = np.ones((64, 3), dtype=bool)
    action[-1] = 123.0
    mask[-1] = False

    validate_discrete_action_target(
        action,
        action_mask=mask,
        action_horizon=64,
        action_dim=3,
        action_class_values=[-1, 0, 1],
        action_target_format="values",
    )

    mask[-1] = True
    with pytest.raises(ValueError, match="No discretization threshold"):
        validate_discrete_action_target(
            action,
            action_mask=mask,
            action_horizon=64,
            action_dim=3,
            action_class_values=[-1, 0, 1],
            action_target_format="values",
        )


def test_class_indices_format_uses_configured_mapping_size():
    action = np.tile(np.asarray([0, 1, 2]), (64, 1)).astype(np.float32)
    mask = np.ones_like(action, dtype=bool)

    validate_discrete_action_target(
        action,
        action_mask=mask,
        action_horizon=64,
        action_dim=3,
        action_class_values=[1, 0, -1],
        action_target_format="class_indices",
    )

    action[0, 0] = 1.5
    with pytest.raises(ValueError, match="integer indices"):
        validate_discrete_action_target(
            action,
            action_mask=mask,
            action_horizon=64,
            action_dim=3,
            action_class_values=[1, 0, -1],
            action_target_format="class_indices",
        )


def test_mapping_must_be_explicit_distinct_three_classes():
    np.testing.assert_array_equal(
        validate_discrete_action_mapping([-1, 0, 1]),
        [-1, 0, 1],
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_discrete_action_mapping([-1, -1, 1])
    np.testing.assert_array_equal(
        validate_discrete_action_mapping([1, -1, 0]),
        [1, -1, 0],
    )
    with pytest.raises(ValueError, match="permutation"):
        validate_discrete_action_mapping([-2, 0, 2])


def test_all_action_keys_must_share_the_same_window():
    delta_indices = {
        "action.x": np.arange(64),
        "action.y": np.arange(64),
    }
    np.testing.assert_array_equal(
        get_shared_action_delta_indices(delta_indices, ["action.x", "action.y"]),
        np.arange(64),
    )

    delta_indices["action.y"] = np.arange(1, 65)
    with pytest.raises(ValueError, match="identical delta indices"):
        get_shared_action_delta_indices(delta_indices, ["action.x", "action.y"])


def test_unregistered_dataset_keys_are_fully_config_driven():
    config = _configured_modality_config(
        {
            "action_horizon": 64,
            "modality_keys": {
                "video": ["video.custom_scope"],
                "state": ["state.custom_offset"],
                "action": ["action.custom_axis"],
                "language": ["annotation.custom.task"],
            },
        }
    )

    assert config["video"].modality_keys == ["video.custom_scope"]
    assert config["action"].modality_keys == ["action.custom_axis"]
    assert config["action"].delta_indices == list(range(64))


def test_discrete_path_keeps_tensor_conversion_but_drops_action_normalization():
    to_tensor = StateActionToTensor(apply_to=["action.axis"])
    normalize = StateActionTransform(
        apply_to=["action.axis"],
        normalization_modes={"action.axis": "q99"},
    )
    filtered = _without_action_transforms(
        ComposedModalityTransform(transforms=[to_tensor, normalize]),
        ["action.axis"],
    )

    assert filtered.transforms == [to_tensor]


def test_mixed_state_action_semantic_transform_is_rejected():
    mixed_normalize = StateActionTransform(
        apply_to=["state.offset", "action.axis"],
        normalization_modes={"action.axis": "q99"},
    )

    with pytest.raises(ValueError, match="mixes discrete action keys"):
        _without_action_transforms(
            ComposedModalityTransform(transforms=[mixed_normalize]),
            ["action.axis"],
        )


def test_dataset_semantics_are_derived_from_discrete_action_head():
    cfg = SimpleNamespace(
        framework=SimpleNamespace(
            action_model={
                "action_head_type": "discrete",
                "action_horizon": 64,
                "action_dim": 3,
                "action_class_values": [1, 0, -1],
                "action_target_format": "class_indices",
            }
        )
    )
    data_cfg = {}

    synchronize_discrete_action_data_config(cfg, data_cfg)

    assert data_cfg["discrete_actions"] is True
    assert data_cfg["action_delta_indices"] == list(range(64))
    assert data_cfg["action_class_values"] == [1, 0, -1]
    assert data_cfg["action_target_format"] == "class_indices"


def test_explicit_dataset_mapping_cannot_drift_from_action_head():
    cfg = SimpleNamespace(
        framework=SimpleNamespace(
            action_model={
                "action_head_type": "discrete",
                "action_horizon": 64,
                "action_dim": 3,
                "action_class_values": [-1, 0, 1],
                "action_target_format": "values",
            }
        )
    )
    data_cfg = {"action_class_values": [1, 0, -1]}

    with pytest.raises(ValueError, match="conflicts"):
        synchronize_discrete_action_data_config(cfg, data_cfg)


def test_endowam_pseudo_z60_uses_the_drive_schema_for_all_subsets():
    mixture = DATASET_NAMED_MIXTURES["endowam_pseudo_z60"]
    assert mixture == [
        ("ureter", 1.0, "endowam_endoscope"),
        ("ercp", 1.0, "endowam_endoscope"),
        ("esophagus", 1.0, "endowam_endoscope"),
    ]

    robot_config = ROBOT_TYPE_CONFIG_MAP["endowam_endoscope"]
    modalities = robot_config.modality_config()
    assert modalities["video"].modality_keys == ["video.endoscope"]
    assert modalities["state"].modality_keys == ["state.endoscope_state"]
    assert modalities["action"].modality_keys == ["action.endoscope_cmd"]
    assert modalities["action"].delta_indices == list(range(64))
    assert modalities["language"].modality_keys == [
        "annotation.human.action.task_description"
    ]
    assert all(
        isinstance(transform, StateActionToTensor)
        for transform in robot_config.transform().transforms
    )
    assert (
        ROBOT_TYPE_TO_EMBODIMENT_TAG["endowam_endoscope"]
        is EmbodimentTag.NEW_EMBODIMENT
    )
