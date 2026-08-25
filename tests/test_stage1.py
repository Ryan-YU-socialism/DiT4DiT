import torch

from DiT4DiT.model.framework.stage1 import (
    latent_alignment_scores,
    mask_action_dimensions,
    repeat_batch,
    select_candidates,
)


def test_repeat_batch_keeps_candidate_groups_contiguous():
    x = torch.tensor([[1], [2]])
    assert repeat_batch(x, 3).tolist() == [[1], [1], [1], [2], [2], [2]]


def test_latent_alignment_and_selection():
    reference = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    candidates = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]]]
    )
    scores = latent_alignment_scores(candidates, reference, num_candidates=2)
    actions = torch.arange(8.0).view(2, 2, 1, 2)
    selected, indices = select_candidates(actions, scores)
    assert indices.tolist() == [0, 1]
    assert torch.equal(selected, torch.stack([actions[0, 0], actions[1, 1]]))


def test_selection_rejects_mismatched_shapes():
    actions = torch.zeros(2, 3, 4, 5)
    scores = torch.zeros(2, 2)
    try:
        select_candidates(actions, scores)
    except ValueError as exc:
        assert "dimensions differ" in str(exc)
    else:
        raise AssertionError("expected shape validation failure")


def test_mask_action_dimensions_zeros_padding():
    actions = torch.ones(1, 2, 3, 4)
    masked = mask_action_dimensions(actions, torch.tensor([True, True, False, False]))
    assert torch.equal(masked[..., :2], torch.ones_like(masked[..., :2]))
    assert torch.equal(masked[..., 2:], torch.zeros_like(masked[..., 2:]))


def test_mask_action_dimensions_rejects_wrong_width():
    try:
        mask_action_dimensions(torch.ones(1, 2, 3), torch.tensor([True, False]))
    except ValueError as exc:
        assert "width D" in str(exc)
    else:
        raise AssertionError("expected action mask width validation failure")
