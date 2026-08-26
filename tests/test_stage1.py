import torch

from DiT4DiT.model.framework.stage1 import (
    latent_alignment_scores,
    mask_action_dimensions,
    repeat_batch,
    resolve_world_model_generator,
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


def test_resolve_world_model_generator_same_seed_gives_identical_noise():
    # This is the property Stage 1 relies on: the reference future and every
    # candidate future are drawn with the *same* fixed_seed and no explicit
    # generator, so their noise draws must match bit-for-bit.
    reference_generator = resolve_world_model_generator(42, None, "cpu")
    candidate_generator = resolve_world_model_generator(42, None, "cpu")
    assert reference_generator is not candidate_generator  # independently constructed
    reference_noise = torch.randn(4, 4, generator=reference_generator)
    candidate_noise = torch.randn(4, 4, generator=candidate_generator)
    assert torch.equal(reference_noise, candidate_noise)


def test_resolve_world_model_generator_different_seeds_diverge():
    reference_generator = resolve_world_model_generator(42, None, "cpu")
    candidate_generator = resolve_world_model_generator(43, None, "cpu")
    reference_noise = torch.randn(4, 4, generator=reference_generator)
    candidate_noise = torch.randn(4, 4, generator=candidate_generator)
    assert not torch.equal(reference_noise, candidate_noise)


def test_resolve_world_model_generator_no_seed_passes_through():
    assert resolve_world_model_generator(None, None, "cpu") is None


def test_resolve_world_model_generator_explicit_generator_wins():
    explicit = torch.Generator(device="cpu").manual_seed(7)
    assert resolve_world_model_generator(42, explicit, "cpu") is explicit
