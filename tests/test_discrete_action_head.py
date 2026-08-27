from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from DiT4DiT.model.modules.action_model.discrete_action_head import (
    DiscreteActionHead,
    decode_discrete_action_targets,
)


def _config(
    *,
    class_values=(-1, 0, 1),
    target_format="values",
    action_horizon=64,
    future_window=63,
):
    action_config = SimpleNamespace(
        action_model_type="DiT-Test",
        hidden_size=16,
        add_pos_embed=True,
        max_seq_len=128,
        action_dim=3,
        state_dim=3,
        future_action_window_size=future_window,
        action_horizon=action_horizon,
        num_action_classes=len(class_values),
        action_class_values=list(class_values),
        action_target_format=target_format,
        discrete_hidden_size=8,
        discrete_num_attention_heads=2,
        discrete_num_layers=1,
        discrete_ffn_dim=16,
        discrete_dropout=0.0,
        vl_embedding_dim=8,
        diffusion_model_cfg={
            "cross_attention_dim": 8,
            "dropout": 0.0,
            "final_dropout": False,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 1,
            "output_dim": 8,
            "positional_embeddings": None,
        },
    )
    return SimpleNamespace(framework=SimpleNamespace(action_model=action_config))


def _make_head(monkeypatch, **config_overrides):
    del monkeypatch
    return DiscreteActionHead(_config(**config_overrides))


def _replace_logits(monkeypatch, head, logits: torch.Tensor):
    def fixed_forward_logits(self, vl_embs, state=None, encoder_attention_mask=None):
        del self, state, encoder_attention_mask
        return logits.to(device=vl_embs.device).expand(vl_embs.shape[0], -1, -1, -1)

    monkeypatch.setattr(type(head), "forward_logits", fixed_forward_logits)


def test_default_action_mapping_round_trips(monkeypatch):
    head = _make_head(monkeypatch)
    targets = torch.tensor([[[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]])

    class_indices = head.encode_targets(targets)

    assert class_indices.dtype == torch.long
    assert class_indices.tolist() == [[[0, 1, 2], [2, 0, 1]]]
    assert torch.equal(head.decode_class_indices(class_indices), targets)


def test_custom_class_values_define_mapping_order(monkeypatch):
    head = _make_head(monkeypatch, class_values=(1, -1, 0))
    targets = torch.tensor([[[-1.0, 0.0, 1.0]]])

    class_indices = head.encode_targets(targets)

    assert class_indices.tolist() == [[[1, 2, 0]]]
    assert torch.equal(head.decode_class_indices(class_indices), targets)


def test_class_index_targets_use_the_checkpoint_mapping(monkeypatch):
    head = _make_head(
        monkeypatch,
        class_values=(1, -1, 0),
        target_format="class_indices",
    )
    class_indices = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float16)

    assert head.encode_targets(class_indices).tolist() == [[[0, 1, 2]]]
    assert head.targets_to_values(class_indices).tolist() == [[[1.0, -1.0, 0.0]]]


def test_masked_padding_decodes_to_neutral_zero(monkeypatch):
    head = _make_head(monkeypatch, target_format="class_indices")
    targets = torch.full((1, 2, 3), 99.0)
    valid_mask = torch.zeros_like(targets)
    targets[0, 0] = torch.tensor([0.0, 1.0, 2.0])
    valid_mask[0, 0] = 1

    values = head.targets_to_values(targets, valid_mask=valid_mask)

    assert values[0, 0].tolist() == [-1.0, 0.0, 1.0]
    assert torch.equal(values[0, 1], torch.zeros(3))


def test_target_codec_works_without_a_policy_instance_for_video_only_training():
    class_indices = torch.tensor([[[0.0, 1.0, 2.0], [99.0, 99.0, 99.0]]])
    valid_mask = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.bool)

    values = decode_discrete_action_targets(
        class_indices,
        action_class_values=[1, -1, 0],
        action_target_format="class_indices",
        valid_mask=valid_mask,
    )

    assert values.tolist() == [[[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]]


def test_forward_logits_are_independent_three_class_predictions_per_axis(monkeypatch):
    head = _make_head(monkeypatch)
    vl_embs = torch.randn(2, 5, 8)
    state = torch.randn(2, 1, 3)

    logits = head.forward_logits(vl_embs, state)

    assert logits.shape == (2, 64, 3, 3)


def test_forward_accepts_float16_targets_and_computes_masked_ce(monkeypatch):
    head = _make_head(monkeypatch)
    logits = torch.linspace(-1.5, 1.5, 64 * 3 * 3).reshape(1, 64, 3, 3)
    _replace_logits(monkeypatch, head, logits)

    targets = torch.zeros(1, 64, 3, dtype=torch.float16)
    targets[0, 0] = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float16)
    targets[0, 1] = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float16)
    action_mask = torch.zeros_like(targets)
    action_mask[0, 0] = 1
    action_mask[0, 1, :2] = 1

    loss = head(
        torch.randn(1, 4, 8),
        targets,
        action_mask,
    )

    target_indices = head.encode_targets(targets)
    valid = action_mask.bool()
    expected = F.cross_entropy(logits[valid], target_indices[valid])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.allclose(loss.float(), expected.float())


def test_masked_invalid_target_is_ignored_before_encoding(monkeypatch):
    head = _make_head(monkeypatch)
    logits = torch.zeros(1, 64, 3, 3)
    _replace_logits(monkeypatch, head, logits)

    targets = torch.zeros(1, 64, 3, dtype=torch.float16)
    targets[0, 7, 2] = 0.25
    action_mask = torch.ones_like(targets)
    action_mask[0, 7, 2] = 0

    loss = head(torch.randn(1, 4, 8), targets, action_mask)

    assert torch.isfinite(loss)
    assert torch.allclose(loss.float(), torch.tensor(3.0).log())


def test_unmasked_non_class_value_is_rejected(monkeypatch):
    head = _make_head(monkeypatch)
    logits = torch.zeros(1, 64, 3, 3)
    _replace_logits(monkeypatch, head, logits)
    targets = torch.zeros(1, 64, 3)
    targets[0, 7, 2] = 0.25

    with pytest.raises(ValueError):
        head(torch.randn(1, 4, 8), targets, torch.ones_like(targets))


def test_predict_action_decodes_argmax_to_configured_values(monkeypatch):
    head = _make_head(monkeypatch)
    class_indices = torch.arange(64 * 3).reshape(1, 64, 3) % 3
    logits = torch.full((1, 64, 3, 3), -10.0)
    logits.scatter_(-1, class_indices.unsqueeze(-1), 10.0)
    _replace_logits(monkeypatch, head, logits)

    actions = head.predict_action(torch.randn(2, 4, 8))

    expected = head.decode_class_indices(class_indices).expand(2, -1, -1)
    assert actions.shape == (2, 64, 3)
    assert torch.equal(actions, expected)
    assert set(actions.unique().tolist()) == {-1.0, 0.0, 1.0}


def test_multiple_candidates_are_decoded_per_axis(monkeypatch):
    head = _make_head(monkeypatch)
    logits = torch.zeros(1, 64, 3, 3)
    _replace_logits(monkeypatch, head, logits)

    actions = head.predict_action(
        torch.randn(2, 4, 8),
        num_candidates=4,
        generator=torch.Generator().manual_seed(7),
    )

    assert actions.shape == (2, 4, 64, 3)
    assert set(actions.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_checkpoint_rejects_a_different_class_order(monkeypatch):
    original = _make_head(monkeypatch, class_values=(-1, 0, 1))
    remapped = _make_head(monkeypatch, class_values=(1, 0, -1))

    with pytest.raises(RuntimeError, match="conflicts with action_class_values"):
        remapped.load_state_dict(original.state_dict())


def test_constructor_rejects_inconsistent_horizon(monkeypatch):
    del monkeypatch

    with pytest.raises(ValueError, match="horizon"):
        DiscreteActionHead(_config(action_horizon=63, future_window=63))
