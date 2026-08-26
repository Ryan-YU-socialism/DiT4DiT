"""Stage-1 candidate planning utilities.

The helpers in this module deliberately do not depend on Cosmos.  Keeping the
candidate reshaping and selection logic here makes the planner testable with a
small fake backbone and also makes it possible to replace the Stage-1 scorer
without changing the policy.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F


@dataclass
class Stage1Output:
    """Result of candidate generation, prediction, and selection."""

    selected_actions: torch.Tensor
    selected_indices: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_scores: torch.Tensor
    predicted_future_videos: Optional[torch.Tensor] = None


def repeat_batch(x: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    """Repeat every batch item consecutively, matching candidate flattening."""
    if x is None:
        return None
    return x.repeat_interleave(repeats, dim=0)


def latent_alignment_scores(
    candidate_latents: torch.Tensor,
    reference_latents: torch.Tensor,
    num_candidates: int,
) -> torch.Tensor:
    """Score candidate futures by cosine alignment with a task-reference future.

    Args:
        candidate_latents: Flattened ``(B*K, S, D)`` predicted-future latents.
        reference_latents: ``(B, S, D)`` task-conditioned reference latents.
        num_candidates: Number of candidates per observation.
    Returns:
        A ``(B, K)`` tensor; larger values are preferred.
    """
    if candidate_latents.ndim != 3 or reference_latents.ndim != 3:
        raise ValueError("candidate_latents and reference_latents must be (B,S,D)")
    batch_size = reference_latents.shape[0]
    if candidate_latents.shape[0] != batch_size * num_candidates:
        raise ValueError("candidate latent batch must equal B * num_candidates")
    candidate_pooled = candidate_latents.float().mean(dim=1)
    reference_pooled = reference_latents.float().mean(dim=1)
    reference_pooled = repeat_batch(reference_pooled, num_candidates)
    scores = F.cosine_similarity(candidate_pooled, reference_pooled, dim=-1)
    return scores.view(batch_size, num_candidates)


def select_candidates(candidate_actions: torch.Tensor, scores: torch.Tensor):
    """Select one action trajectory per batch from ``(B,K,T,D)`` candidates."""
    if candidate_actions.ndim != 4 or scores.ndim != 2:
        raise ValueError("candidate_actions must be (B,K,T,D) and scores must be (B,K)")
    if candidate_actions.shape[:2] != scores.shape:
        raise ValueError("candidate action and score batch/candidate dimensions differ")
    indices = scores.argmax(dim=1)
    batch_indices = torch.arange(candidate_actions.shape[0], device=candidate_actions.device)
    return candidate_actions[batch_indices, indices], indices


def resolve_world_model_generator(
    fixed_seed: Optional[int],
    generator: Optional[Union[torch.Generator, "list[torch.Generator]"]],
    device: Union[str, torch.device],
) -> Optional[Union[torch.Generator, "list[torch.Generator]"]]:
    """Build the world-model noise generator used by both the reference and
    every candidate rollout in Stage 1.

    An explicit ``generator`` always wins. Otherwise, a fresh ``torch.Generator``
    is seeded from ``fixed_seed``. Reference and candidate calls pass the same
    ``fixed_seed`` and no explicit ``generator``, so each independently builds a
    generator with identical state -- this is what makes their noise draws
    match and Stage-1 scores reflect the action conditioning rather than
    sampling variance. Kept as a standalone function so that invariant can be
    unit-tested without constructing the Cosmos extractor.
    """
    if fixed_seed is not None and generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(fixed_seed)
    return generator


def mask_action_dimensions(
    candidate_actions: torch.Tensor,
    action_dimension_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Zero padded action dimensions before world-model conditioning."""
    if action_dimension_mask is None:
        return candidate_actions
    mask = torch.as_tensor(
        action_dimension_mask,
        device=candidate_actions.device,
        dtype=torch.bool,
    )
    if mask.ndim != 1 or mask.shape[0] != candidate_actions.shape[-1]:
        raise ValueError("action_dimension_mask must be one-dimensional with width D")
    return candidate_actions.masked_fill(~mask, 0)
