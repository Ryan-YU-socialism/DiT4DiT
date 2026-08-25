# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Teli Ma/ HKUST GZ] in [2025]. 


import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.framework.stage1 import (
    Stage1Output,
    latent_alignment_scores,
    mask_action_dimensions,
    select_candidates,
)
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead, get_action_model
from DiT4DiT.model.modules.vlm import get_backbone_model
from DiT4DiT.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("DiT4DiT")
class DiT4DiT(baseframework):
    """
    Multimodal vision-language-action model with Stage-1 candidate planning.

    Components:
      - Cosmos-Predict2.5 world-model backbone
      - DiT flow-matching head for action sequence modeling
      - Optional action-conditioned future rollout and latent selector

    Focus: predict and rank future continuous action plans conditioned on images
    and task instructions.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config

        # Determine training mode from config: "video", "action", or "joint"
        training_mode = config.framework.cosmos25.training.lower() if config is not None else "action"
        self.video_fm_only = (training_mode == "video")
        stage1_cfg = getattr(config.framework, "stage1", None) if config is not None else None
        self.stage1_enabled = bool(getattr(stage1_cfg, "enabled", False))
        future_loss_type = (
            str(getattr(config.framework.cosmos25, "future_loss_type", "")).lower()
            if config is not None
            else ""
        )
        supported_stage1_losses = {
            "flow_matching",
            "latent_flow_matching",
            "rectified_flow",
            "rf",
        }
        if self.stage1_enabled and future_loss_type not in supported_stage1_losses:
            raise ValueError(
                "Stage 1 action conditioning currently requires a latent flow-matching "
                f"future loss, got {future_loss_type!r}."
            )

        self.backbone_interface = get_backbone_model(config=self.config)

        # -------- Align DiT cross-attention dim with backbone output dim --------
        # GR00T ActionHead uses `diffusion_model_cfg.cross_attention_dim` to match vl_embs' last dim.
        vl_hidden_dim = None
        if hasattr(self.backbone_interface, "model") and hasattr(self.backbone_interface.model, "config"):
            vl_hidden_dim = getattr(self.backbone_interface.model.config, "hidden_size", None)
        if vl_hidden_dim is None and hasattr(self.backbone_interface, "extractor"):
            vl_hidden_dim = getattr(self.backbone_interface.extractor, "hidden_size", None)
        if vl_hidden_dim is None:
            vl_hidden_dim = getattr(self.config.framework.cosmos25, "vl_hidden_dim", None)

        if not self.video_fm_only:
            if vl_hidden_dim is None:
                raise ValueError(
                    "Cannot infer `vl_hidden_dim` for the selected backbone. "
                    "Please set `framework.cosmos25.vl_hidden_dim` in your config."
                )
            self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

            self.future_action_window_size = config.framework.action_model.future_action_window_size
            self.past_action_window_size = config.framework.action_model.past_action_window_size
            self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        else:
            # Video-only mode: skip action model entirely
            self.action_model = None
            self.future_action_window_size = 0
            self.past_action_window_size = 0
            self.chunk_len = 0
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B, [frame_0, frame_1, ..., frame_T-1]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples] if "action" in examples[0] else None
        action_condition = None
        if actions is not None and self.stage1_enabled:
            action_condition = torch.as_tensor(np.asarray(actions), device=self.device)
            action_horizon = int(self.config.framework.action_model.future_action_window_size) + 1
            action_condition = action_condition[:, -action_horizon:]

        # Step 1: backbone input format
        # All video frames (condition + future) are already in batch_images;
        # build_cosmos_inputs splits them into videos (cond) and future_videos internally.
        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)
        backbone_inputs["action_condition"] = action_condition
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            if not self.video_fm_only:
                last_hidden = backbone_outputs.hidden_states[-1]  # [B, L, H] ##2560-4b
            else:
                last_hidden = None
            future_video_loss = getattr(backbone_outputs, "future_video_loss", None)

        # Video-only FM training: no action branch.
        if self.video_fm_only:
            if future_video_loss is None:
                raise ValueError(
                    "video_fm_only is enabled (cosmos25.training='video') but `future_video_loss` is None. "
                    "Please provide `image_next` (or future_images) and set "
                    "`framework.cosmos25.future_loss_type=flow_matching`."
                )
            return {"future_video_loss": future_video_loss}

        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        action_mask = [example["action_mask"] for example in examples]  # [B, len, action_dim]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            action_mask = torch.from_numpy(np.stack(action_mask)).to(last_hidden.device)
            action_mask = action_mask.repeat(repeated_diffusion_steps, 1, 1)
            ###no state
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                action_mask,
                state_repeated,
            )

        out = {"action_loss": action_loss}
        if future_video_loss is not None:
            out["future_video_loss"] = future_video_loss
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Steps:
          1. Route to Stage 1 when enabled
          2. Otherwise encode the observation and sample one action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if self.stage1_enabled and not kwargs.pop("disable_stage1", False):
            return self.predict_action_stage1(
                examples,
                num_candidates=kwargs.pop("num_candidates", None),
            )
        if type(examples) is not list:
            examples = [examples]
        batch_images = []
        for ex in examples:
            img = ex["image"]
            if isinstance(img, (list, tuple)) and len(img) > 0:
                batch_images.append(img)
            else:
                batch_images.append([img])
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
    
        # Step 1: backbone input format
        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = backbone_outputs.hidden_states[-1]   # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(
                last_hidden.device,
                dtype=last_hidden.dtype,
            )
            if state is not None
            else None
        )
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_stage1(self, examples: List[dict], num_candidates: Optional[int] = None):
        """Run the Stage-1 FOREWARN loop and return the best of K action plans.

        Candidate futures are predicted by the action-conditioned Cosmos world
        model.  Its latent alignment with a task-conditioned reference future is
        used as the zero-shot VLM score.  Reference and candidates share the
        world-model noise seed so scores measure action effects rather than noise.
        """
        if not self.stage1_enabled:
            raise RuntimeError("Stage 1 is disabled; set framework.stage1.enabled=true")
        if not isinstance(examples, list):
            examples = [examples]
        stage1_cfg = getattr(self.config.framework, "stage1", None)
        if num_candidates is None:
            num_candidates = int(getattr(stage1_cfg, "num_candidates", 4))
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")

        # Dataset examples may include future supervision frames.  Planning must
        # only consume the current observation, just like online deployment.
        batch_images = []
        for example in examples:
            images = example["image"]
            if isinstance(images, (list, tuple)) and not images:
                raise ValueError("Each Stage 1 example must contain a current image")
            current_image = images[0] if isinstance(images, (list, tuple)) else images
            batch_images.append([current_image])
        instructions = [ex["lang"] for ex in examples]
        base_inputs = self.backbone_interface.build_cosmos_inputs(batch_images, instructions)
        policy_out = self.backbone_interface(**base_inputs, return_dict=True)
        policy_latents = policy_out.hidden_states[-1]
        reference_out = self.backbone_interface(
            **base_inputs,
            predict_future=True,
            return_dict=True,
        )
        reference_latents = reference_out.hidden_states[-1]

        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        state_tensor = (
            torch.as_tensor(
                np.asarray(state),
                device=policy_latents.device,
                dtype=policy_latents.dtype,
            )
            if state is not None
            else None
        )
        candidates = self.action_model.predict_action(
            policy_latents,
            state_tensor,
            num_candidates=num_candidates,
        )
        if num_candidates == 1:
            candidates = candidates.unsqueeze(1)

        # Padded dimensions receive no action loss and otherwise remain random.
        # Zero them before conditioning the world model, matching the training data.
        valid_action_dim = int(
            getattr(stage1_cfg, "valid_action_dim", candidates.shape[-1])
        )
        if not 1 <= valid_action_dim <= candidates.shape[-1]:
            raise ValueError(
                "framework.stage1.valid_action_dim must be between 1 and action_dim"
            )
        action_dimension_mask = torch.arange(
            candidates.shape[-1],
            device=candidates.device,
        ) < valid_action_dim
        candidates = mask_action_dimensions(candidates, action_dimension_mask)

        flat_candidates = candidates.flatten(0, 1)
        repeated_images = [frames for frames in batch_images for _ in range(num_candidates)]
        repeated_instructions = [text for text in instructions for _ in range(num_candidates)]
        world_inputs = self.backbone_interface.build_cosmos_inputs(repeated_images, repeated_instructions)
        world_inputs["action_condition"] = flat_candidates
        world_out = self.backbone_interface(**world_inputs, predict_future=True, return_dict=True)
        scores = latent_alignment_scores(world_out.hidden_states[-1], reference_latents, num_candidates)
        selected, indices = select_candidates(candidates, scores)

        future_videos = world_out.pred_future_video
        if future_videos is not None:
            future_videos = future_videos.view(len(examples), num_candidates, *future_videos.shape[1:])
        result = Stage1Output(selected, indices, candidates, scores, future_videos)
        output = {
            "normalized_actions": result.selected_actions.detach().cpu().numpy(),
            "selected_indices": result.selected_indices.detach().cpu().numpy(),
            "candidate_actions": result.candidate_actions.detach().cpu().numpy(),
            "candidate_scores": result.candidate_scores.detach().cpu().numpy(),
        }
        if result.predicted_future_videos is not None:
            output["predicted_future_videos"] = result.predicted_future_videos.detach().cpu().numpy()
        return output
