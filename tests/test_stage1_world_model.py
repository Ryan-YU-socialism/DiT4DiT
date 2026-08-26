"""Integration-level Stage-1 tests.

test_stage1.py only exercises the pure scoring/selection helpers with hand-built
tensors. It does not touch the two places where a refactor could silently break
the FOREWARN safety property ("reference and candidate futures share noise, and
the reference never sees a candidate's action"):

  1. `_Cosmos25_Interface.forward` -- must forward the *same* `fixed_seed` (and
     the same generate_future/capture_final_hidden/deterministic_conditioning
     flags) to the extractor for both the reference and every candidate
     rollout, and must not silently thread an explicit `generator` that would
     bypass the shared seed.
  2. `DiT4DiT.predict_action_stage1` -- must call the world model for the
     reference future with `action_condition=None`, only pass the real
     (flattened, repeated) candidate actions to the candidate rollout, and
     correctly route the highest-scoring candidate's *own* action trajectory
     back out per batch item.

Both are covered here against lightweight fakes (no Cosmos weights, no GPU)
by bypassing `__init__` and only wiring the attributes each method actually
reads -- this is intentionally a white-box test of the plumbing, not a
behavioral test of the world model itself.
"""

from types import SimpleNamespace

import numpy as np
import torch

from DiT4DiT.model.framework.DiT4DiT import DiT4DiT
from DiT4DiT.model.modules.vlm.Cosmos25 import _Cosmos25_Interface


class _RecordingExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["videos"].shape[0]
        hidden = torch.zeros(batch_size, 1, 2)
        # The interface only unpacks a 3-tuple on the predict_future/future_videos
        # branch; the plain-encode branch calls the extractor for a bare tensor.
        if "generate_future" in kwargs:
            return hidden, None, None
        return hidden


def _make_interface_stub(extractor, world_model_seed=42):
    stub = _Cosmos25_Interface.__new__(_Cosmos25_Interface)
    stub.config = SimpleNamespace(
        framework=SimpleNamespace(
            cosmos25=SimpleNamespace(
                conditional_frame_timestep=0.001,
                future_num_inference_steps=2,
            ),
            stage1=SimpleNamespace(world_model_seed=world_model_seed),
        )
    )
    stub.extractor = extractor
    return stub


def test_reference_and_candidate_rollouts_get_identical_seed_and_flags():
    extractor = _RecordingExtractor()
    stub = _make_interface_stub(extractor)

    reference_videos = torch.zeros(2, 3, 1, 16, 16)
    stub.forward(
        prompts=["do task a", "do task b"],
        videos=reference_videos,
        height=16,
        width=16,
        predict_future=True,
        action_condition=None,
    )

    candidate_actions = torch.arange(6 * 1 * 4).float().view(6, 1, 4)
    candidate_videos = torch.zeros(6, 3, 1, 16, 16)
    stub.forward(
        prompts=["do task a"] * 3 + ["do task b"] * 3,
        videos=candidate_videos,
        height=16,
        width=16,
        predict_future=True,
        action_condition=candidate_actions,
    )

    assert len(extractor.calls) == 2
    reference_call, candidate_call = extractor.calls

    # The property Stage 1 depends on: same seed, no explicit generator passed
    # by either caller, so the extractor independently seeds identical noise.
    assert reference_call["fixed_seed"] == candidate_call["fixed_seed"] == 42
    assert "generator" not in reference_call and "generator" not in candidate_call
    for call in (reference_call, candidate_call):
        assert call["generate_future"] is True
        assert call["capture_final_hidden"] is True
        assert call["deterministic_conditioning"] is True

    # Only the candidate call may see real actions.
    assert reference_call["action_condition"] is None
    assert candidate_call["action_condition"] is candidate_actions


def test_non_future_call_disables_seed_and_determinism_machinery():
    # The ordinary (training / policy) path must not accidentally pick up
    # Stage-1's determinism, since that is only meant for the world-model
    # rollouts that get compared against each other.
    extractor = _RecordingExtractor()
    stub = _make_interface_stub(extractor)

    stub.forward(
        prompts=["do task a"],
        videos=torch.zeros(1, 3, 1, 16, 16),
        height=16,
        width=16,
        predict_future=False,
    )

    (call,) = extractor.calls
    # The plain-encode branch doesn't even pass these kwargs -- Stage 1's
    # seed-sharing/determinism machinery is exclusive to the future-rollout path.
    for key in ("fixed_seed", "generate_future", "capture_final_hidden", "deterministic_conditioning"):
        assert key not in call, f"{key!r} should not reach the extractor on the non-future path"


class _FakeBackbone:
    """Stands in for `backbone_interface`: records every call and returns
    caller-controlled latents keyed only on whether `action_condition` was
    supplied, so tests can assert on *which* inputs reached the world model
    without needing a real Cosmos forward pass."""

    def __init__(self, policy_latents, reference_latents, candidate_latents):
        self.policy_latents = policy_latents
        self.reference_latents = reference_latents
        self.candidate_latents = candidate_latents
        self.calls = []

    def build_cosmos_inputs(self, images, instructions):
        return {"images": images, "instructions": instructions}

    def __call__(self, *, images, instructions, predict_future=False, action_condition=None, return_dict=True):
        self.calls.append(
            {
                "n_examples": len(images),
                "predict_future": predict_future,
                "action_condition": action_condition,
            }
        )
        if action_condition is not None:
            hidden = self.candidate_latents
        elif predict_future:
            hidden = self.reference_latents
        else:
            hidden = self.policy_latents
        return SimpleNamespace(hidden_states=[hidden], pred_future_video=None)


class _FakeActionModel:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def predict_action(self, policy_latents, state, num_candidates):
        self.calls.append({"num_candidates": num_candidates})
        return self.candidates


def _make_stage1_stub(backbone, action_model, num_candidates=3, valid_action_dim=4):
    stub = DiT4DiT.__new__(DiT4DiT)
    stub.config = SimpleNamespace(
        framework=SimpleNamespace(
            stage1=SimpleNamespace(num_candidates=num_candidates, valid_action_dim=valid_action_dim)
        )
    )
    stub.stage1_enabled = True
    stub.backbone_interface = backbone
    stub.action_model = action_model
    return stub


def test_predict_action_stage1_never_leaks_actions_into_reference_call():
    num_candidates = 3
    # Two batch items, 2D latents (S=1) so cosine similarity reduces to comparing
    # these vectors directly. Candidate index 2 is the best match for batch 0's
    # reference [1, 0]; candidate index 0 is the best match for batch 1's
    # reference [0, 1].
    reference_latents = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    candidate_latents = torch.tensor(
        [
            [[0.0, 1.0]],  # batch 0, candidate 0
            [[-1.0, 0.0]],  # batch 0, candidate 1
            [[1.0, 0.0]],  # batch 0, candidate 2 (best match)
            [[0.0, 1.0]],  # batch 1, candidate 0 (best match)
            [[1.0, 0.0]],  # batch 1, candidate 1
            [[-1.0, 0.0]],  # batch 1, candidate 2
        ]
    )
    candidate_actions = torch.arange(2 * num_candidates * 1 * 4).float().view(2, num_candidates, 1, 4)

    backbone = _FakeBackbone(
        policy_latents=reference_latents,
        reference_latents=reference_latents,
        candidate_latents=candidate_latents,
    )
    action_model = _FakeActionModel(candidate_actions)
    stub = _make_stage1_stub(backbone, action_model, num_candidates=num_candidates, valid_action_dim=4)

    examples = [
        {"image": "img0", "lang": "task a", "state": np.zeros(2, dtype=np.float32)},
        {"image": "img1", "lang": "task b", "state": np.zeros(2, dtype=np.float32)},
    ]
    output = stub.predict_action_stage1(examples, num_candidates=num_candidates)

    assert len(backbone.calls) == 3
    policy_call, reference_call, world_call = backbone.calls

    # The reference future must be built from the raw observation alone.
    assert reference_call["predict_future"] is True
    assert reference_call["action_condition"] is None
    assert reference_call["n_examples"] == 2  # not repeated per-candidate

    # Only the world-model call may see (flattened, repeated) real actions.
    assert world_call["predict_future"] is True
    assert world_call["action_condition"] is not None
    assert world_call["action_condition"].shape == (2 * num_candidates, 1, 4)
    assert torch.equal(world_call["action_condition"], candidate_actions.flatten(0, 1))
    assert world_call["n_examples"] == 2 * num_candidates

    assert action_model.calls == [{"num_candidates": num_candidates}]

    # Selection must route back to *that batch item's own* winning candidate.
    assert output["selected_indices"].tolist() == [2, 0]
    expected_selected = torch.stack([candidate_actions[0, 2], candidate_actions[1, 0]]).numpy()
    assert np.array_equal(output["normalized_actions"], expected_selected)
    assert np.array_equal(output["candidate_actions"], candidate_actions.numpy())


def test_predict_action_stage1_rejects_when_disabled():
    stub = _make_stage1_stub(_FakeBackbone(None, None, None), _FakeActionModel(None))
    stub.stage1_enabled = False
    try:
        stub.predict_action_stage1([{"image": "img0", "lang": "task a"}])
    except RuntimeError as exc:
        assert "Stage 1 is disabled" in str(exc)
    else:
        raise AssertionError("expected predict_action_stage1 to reject a disabled Stage 1")
