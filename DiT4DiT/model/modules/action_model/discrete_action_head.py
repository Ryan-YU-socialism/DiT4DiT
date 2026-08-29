"""Independent per-axis discrete action head for EndoWAM.

The head predicts three categorical distributions for every action timestep,
one for each spatial axis.  It deliberately does not discretize continuous
values with thresholds: the class/value correspondence must be supplied by
``action_class_values`` in the action-model configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


_MISSING = object()


def _unique_for_validation(values: torch.Tensor) -> torch.Tensor:
    """Run unique in a dtype supported consistently on CPU and CUDA."""
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    return torch.unique(values)


def _config_get(config: Any, name: str, default: Any = _MISSING) -> Any:
    """Read a field from a dict, OmegaConf object, or namespace."""

    if isinstance(config, Mapping):
        if name in config:
            return config[name]
    elif hasattr(config, name):
        return getattr(config, name)

    if default is _MISSING:
        raise ValueError(f"Missing required discrete action config field: {name!r}")
    return default


def _action_config(full_config: Any) -> Any:
    """Accept either the global config or the action-model config directly."""

    framework = _config_get(full_config, "framework", None)
    if framework is None:
        return full_config
    return _config_get(framework, "action_model")


def _materialize_config_sequence(value: Any) -> Any:
    """Convert OmegaConf/access-tracker sequence wrappers to plain lists."""

    if torch.is_tensor(value) or isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Mapping):
        return value
    try:
        return [_materialize_config_sequence(item) for item in value]
    except TypeError:
        return value


def decode_discrete_action_targets(
    actions: torch.Tensor,
    *,
    action_class_values: Sequence[float | int] | torch.Tensor,
    action_target_format: str,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode dataset targets to semantic ``-1/0/+1`` commands.

    This codec is intentionally independent of the policy module so the
    action-conditioned world model can also train in video-only mode.  It uses
    exact lookup and never converts continuous offsets with an inferred
    threshold.
    """

    if not torch.is_tensor(actions):
        raise TypeError(f"actions must be a torch.Tensor, got {type(actions)}")
    if actions.ndim < 1 or actions.shape[-1] != 3:
        raise ValueError(
            f"Discrete actions must end in three axes, got shape {tuple(actions.shape)}"
        )
    if actions.dtype == torch.bool:
        raise TypeError("Discrete actions cannot have boolean dtype")

    if valid_mask is None:
        mask = torch.ones_like(actions, dtype=torch.bool)
    else:
        if not torch.is_tensor(valid_mask) or valid_mask.shape != actions.shape:
            raise ValueError("valid_mask must be a tensor with the same shape as actions")
        if valid_mask.device != actions.device:
            raise ValueError("valid_mask and actions must be on the same device")
        if valid_mask.is_floating_point() and not torch.isfinite(valid_mask).all():
            raise ValueError("valid_mask must not contain NaN or infinity")
        if not torch.all((valid_mask == 0) | (valid_mask == 1)):
            raise ValueError("valid_mask must contain only boolean/0/1 values")
        mask = valid_mask.bool()

    class_values = torch.as_tensor(
        _materialize_config_sequence(action_class_values),
        device=actions.device,
        dtype=torch.float32,
    )
    if class_values.shape != (3,) or not torch.isfinite(class_values).all():
        raise ValueError("action_class_values must contain three finite values")
    required_values = torch.tensor(
        [-1.0, 0.0, 1.0], device=actions.device, dtype=class_values.dtype
    )
    if not torch.equal(class_values.sort().values, required_values):
        raise ValueError("action_class_values must be a permutation of [-1, 0, 1]")

    target_format = str(action_target_format).lower()
    if target_format == "values":
        values_for_match = class_values.to(dtype=actions.dtype)
        matches = actions.unsqueeze(-1).eq(values_for_match.view(*(1,) * actions.ndim, 3))
        invalid = mask & ~matches.any(dim=-1)
        if invalid.any():
            bad_values = _unique_for_validation(actions[invalid]).detach().cpu().tolist()
            raise ValueError(
                "Every valid discrete action must exactly match action_class_values; "
                f"invalid values include {bad_values[:8]}. No threshold is inferred."
            )
        class_indices = matches.to(torch.int64).argmax(dim=-1)
    elif target_format == "class_indices":
        safe_actions = actions.masked_fill(~mask, 0)
        if safe_actions.is_floating_point():
            valid_actions = safe_actions[mask]
            if not torch.isfinite(valid_actions).all():
                raise ValueError("Valid class-index actions must be finite")
            if not torch.equal(valid_actions, valid_actions.round()):
                raise ValueError("Valid class-index actions must be exact integers")
        class_indices = safe_actions.to(torch.int64)
        invalid = mask & ((class_indices < 0) | (class_indices >= 3))
        if invalid.any():
            bad_values = _unique_for_validation(actions[invalid]).detach().cpu().tolist()
            raise ValueError(
                "Valid class-index actions must be in [0, 2]; "
                f"invalid values include {bad_values[:8]}"
            )
    else:
        raise ValueError(
            "action_target_format must be 'values' or 'class_indices', "
            f"got {target_format!r}"
        )

    class_indices = class_indices.masked_fill(~mask, 0)
    decoded = class_values[class_indices]
    return decoded.masked_fill(~mask, 0)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return parsed


class DiscreteActionHead(nn.Module):
    """Predict independent ``{-1, 0, +1}``-style classes for three axes.

    The semantic set is fixed to ``{-1, 0, +1}``, but its class-id ordering is
    explicit.  For example, ``action_class_values=[-1, 0, 1]`` defines class
    ids 0, 1, and 2 in that order.  This ordering is registered as a persistent
    buffer and checked when a state dict is loaded, so a checkpoint cannot
    silently be interpreted with a different mapping.

    Expected action-model configuration fields:

    * ``action_dim=3``
    * ``future_action_window_size=63``
    * ``action_horizon=64``
    * ``action_class_values``: a permutation of ``[-1, 0, 1]``
    * ``action_target_format``: ``"values"`` or ``"class_indices"``

    Optional architecture fields are prefixed with ``discrete_``.  Class
    weights may be shared across axes (shape ``[3]``) or axis-specific (shape
    ``[3, 3]``).
    """

    NUM_AXES = 3
    NUM_CLASSES = 3
    ACTION_HORIZON = 64
    _TARGET_FORMAT_CODES = {"values": 0, "class_indices": 1}
    is_discrete = True

    def __init__(self, full_config: Any):
        super().__init__()
        config = _action_config(full_config)
        self.full_config = full_config
        self.config = config

        self.action_dim = _positive_int(_config_get(config, "action_dim"), "action_dim")
        if self.action_dim != self.NUM_AXES:
            raise ValueError(
                f"DiscreteActionHead requires action_dim={self.NUM_AXES}, "
                f"got {self.action_dim}"
            )

        future_window = _positive_int(
            _config_get(config, "future_action_window_size"),
            "future_action_window_size",
        )
        configured_horizon = _positive_int(
            _config_get(config, "action_horizon"), "action_horizon"
        )
        derived_horizon = future_window + 1
        if configured_horizon != derived_horizon:
            raise ValueError(
                "Inconsistent action horizon: action_horizon must equal "
                "future_action_window_size + 1, got "
                f"{configured_horizon} and {future_window} + 1"
            )
        if configured_horizon != self.ACTION_HORIZON:
            raise ValueError(
                f"DiscreteActionHead requires a {self.ACTION_HORIZON}-step horizon, "
                f"got {configured_horizon}"
            )
        self.action_horizon = configured_horizon

        configured_num_classes = _config_get(
            config, "num_action_classes", self.NUM_CLASSES
        )
        if _positive_int(configured_num_classes, "num_action_classes") != self.NUM_CLASSES:
            raise ValueError(
                f"DiscreteActionHead requires exactly {self.NUM_CLASSES} classes per axis"
            )

        class_values = self._parse_class_values(
            _config_get(config, "action_class_values")
        )
        self.register_buffer("action_class_values", class_values, persistent=True)

        target_format = str(_config_get(config, "action_target_format")).lower()
        if target_format not in self._TARGET_FORMAT_CODES:
            allowed = ", ".join(sorted(self._TARGET_FORMAT_CODES))
            raise ValueError(
                f"action_target_format must be one of {{{allowed}}}, got {target_format!r}"
            )
        self.action_target_format = target_format

        class_weights = self._parse_class_weights(
            _config_get(config, "action_class_weights", None)
        )
        self.register_buffer("action_class_weights", class_weights, persistent=True)

        model_dim = _positive_int(
            _config_get(config, "discrete_hidden_size", 512),
            "discrete_hidden_size",
        )
        num_heads = _positive_int(
            _config_get(config, "discrete_num_attention_heads", 8),
            "discrete_num_attention_heads",
        )
        num_layers = _positive_int(
            _config_get(config, "discrete_num_layers", 4),
            "discrete_num_layers",
        )
        ffn_dim = _positive_int(
            _config_get(config, "discrete_ffn_dim", 4 * model_dim),
            "discrete_ffn_dim",
        )
        if model_dim % num_heads != 0:
            raise ValueError(
                "discrete_hidden_size must be divisible by "
                "discrete_num_attention_heads, got "
                f"{model_dim} and {num_heads}"
            )

        dropout = float(_config_get(config, "discrete_dropout", 0.1))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"discrete_dropout must be in [0, 1), got {dropout}")

        vl_embedding_dim = self._resolve_vl_embedding_dim(full_config, config)
        state_dim_raw = _config_get(config, "state_dim", 0)
        if isinstance(state_dim_raw, bool):
            raise ValueError(f"state_dim must be a non-negative integer, got {state_dim_raw!r}")
        try:
            state_dim = int(state_dim_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"state_dim must be a non-negative integer, got {state_dim_raw!r}"
            ) from exc
        if state_dim < 0 or state_dim != state_dim_raw:
            raise ValueError(
                f"state_dim must be a non-negative integer, got {state_dim_raw!r}"
            )

        self.model_dim = model_dim
        self.vl_embedding_dim = vl_embedding_dim
        self.state_dim = state_dim

        self.memory_projection = nn.Linear(vl_embedding_dim, model_dim)
        self.state_projection = (
            nn.Linear(state_dim, model_dim) if state_dim > 0 else None
        )
        self.action_queries = nn.Parameter(
            torch.empty(self.action_horizon, model_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # TransformerDecoder self-attention has no causal mask here, so all 64
        # action queries attend bidirectionally; cross-attention reads VLM/state
        # memory.
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.classifier = nn.Linear(
            model_dim, self.action_dim * self.NUM_CLASSES
        )

        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @classmethod
    def _parse_class_values(cls, raw_values: Any) -> torch.Tensor:
        raw_values = _materialize_config_sequence(raw_values)
        if isinstance(raw_values, (str, bytes)):
            raise ValueError(
                "action_class_values must be an explicit sequence of three numeric values"
            )
        try:
            values_list = list(raw_values)
        except TypeError as exc:
            raise ValueError(
                "action_class_values must be an explicit sequence of three numeric values"
            ) from exc
        if len(values_list) != cls.NUM_CLASSES:
            raise ValueError(
                f"action_class_values must contain exactly {cls.NUM_CLASSES} values, "
                f"got {len(values_list)}"
            )
        try:
            values = torch.tensor(values_list, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_class_values must contain only numeric values") from exc
        if values.ndim != 1 or values.numel() != cls.NUM_CLASSES:
            raise ValueError("action_class_values must be a flat sequence of three values")
        if not torch.isfinite(values).all():
            raise ValueError("action_class_values must all be finite")
        if _unique_for_validation(values).numel() != cls.NUM_CLASSES:
            raise ValueError("action_class_values must be pairwise distinct")
        required_values = torch.tensor([-1.0, 0.0, 1.0], dtype=values.dtype)
        if not torch.equal(values.sort().values, required_values):
            raise ValueError(
                "action_class_values must be a permutation of [-1, 0, 1]; "
                "the order defines checkpoint-compatible class ids"
            )
        return values

    @classmethod
    def _parse_class_weights(cls, raw_weights: Any) -> torch.Tensor:
        if raw_weights is None:
            weights = torch.ones(cls.NUM_AXES, cls.NUM_CLASSES, dtype=torch.float32)
        else:
            try:
                weights = torch.as_tensor(
                    _materialize_config_sequence(raw_weights),
                    dtype=torch.float32,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("action_class_weights must be numeric") from exc
            if weights.shape == (cls.NUM_CLASSES,):
                weights = weights.unsqueeze(0).expand(cls.NUM_AXES, -1).clone()
            elif weights.shape != (cls.NUM_AXES, cls.NUM_CLASSES):
                raise ValueError(
                    "action_class_weights must have shape [3] or [3, 3], "
                    f"got {tuple(weights.shape)}"
                )
        if not torch.isfinite(weights).all():
            raise ValueError("action_class_weights must all be finite")
        if (weights < 0).any():
            raise ValueError("action_class_weights must be non-negative")
        if (weights.sum(dim=-1) <= 0).any():
            raise ValueError("each axis must have at least one positive class weight")
        return weights.contiguous()

    @staticmethod
    def _resolve_vl_embedding_dim(full_config: Any, action_config: Any) -> int:
        value = _config_get(action_config, "vl_embedding_dim", None)
        if value is None:
            diffusion_config = _config_get(
                action_config, "diffusion_model_cfg", None
            )
            if diffusion_config is not None:
                value = _config_get(diffusion_config, "cross_attention_dim", None)
        if value is None:
            framework = _config_get(full_config, "framework", None)
            cosmos_config = (
                _config_get(framework, "cosmos25", None)
                if framework is not None
                else None
            )
            if cosmos_config is not None:
                value = _config_get(cosmos_config, "vl_hidden_dim", None)
        if value is None:
            raise ValueError(
                "Missing VLM embedding width: set action_model.vl_embedding_dim, "
                "action_model.diffusion_model_cfg.cross_attention_dim, or "
                "framework.cosmos25.vl_hidden_dim"
            )
        return _positive_int(value, "vl_embedding_dim")

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject checkpoints whose categorical semantics differ from config."""

        values_key = prefix + "action_class_values"
        incoming_values = state_dict.get(values_key)
        if incoming_values is not None:
            expected = self.action_class_values.detach().to(
                device=incoming_values.device, dtype=incoming_values.dtype
            )
            if incoming_values.shape != expected.shape or not torch.equal(
                incoming_values, expected
            ):
                error_msgs.append(
                    f"{values_key} conflicts with action_class_values in the current "
                    f"config: checkpoint={incoming_values.detach().cpu().tolist()}, "
                    f"config={self.action_class_values.detach().cpu().tolist()}"
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_vl_embeddings(self, vl_embs: torch.Tensor) -> None:
        if not torch.is_tensor(vl_embs):
            raise TypeError(f"vl_embs must be a torch.Tensor, got {type(vl_embs)}")
        if vl_embs.ndim != 3:
            raise ValueError(
                f"vl_embs must have shape [B, S, D], got {tuple(vl_embs.shape)}"
            )
        if vl_embs.shape[0] <= 0 or vl_embs.shape[1] <= 0:
            raise ValueError("vl_embs must have non-empty batch and sequence dimensions")
        if vl_embs.shape[-1] != self.vl_embedding_dim:
            raise ValueError(
                f"vl_embs width must be {self.vl_embedding_dim}, "
                f"got {vl_embs.shape[-1]}"
            )
        if not vl_embs.is_floating_point():
            raise TypeError(f"vl_embs must be floating point, got {vl_embs.dtype}")
        if vl_embs.device != self.action_queries.device:
            raise ValueError(
                f"vl_embs is on {vl_embs.device}, but DiscreteActionHead is on "
                f"{self.action_queries.device}"
            )

    @staticmethod
    def _binary_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        if mask.is_floating_point() and not torch.isfinite(mask).all():
            raise ValueError(f"{name} must not contain NaN or infinity")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain only boolean/0/1 values")
        return mask.bool()

    def _memory_padding_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        memory_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if encoder_attention_mask is None:
            return None
        if not torch.is_tensor(encoder_attention_mask):
            raise TypeError("encoder_attention_mask must be a torch.Tensor or None")
        if encoder_attention_mask.shape != (batch_size, memory_length):
            raise ValueError(
                "encoder_attention_mask must have shape [B, S] matching vl_embs, "
                f"expected {(batch_size, memory_length)}, got "
                f"{tuple(encoder_attention_mask.shape)}"
            )
        valid_mask = self._binary_mask(
            encoder_attention_mask.to(device=device), "encoder_attention_mask"
        )
        if not valid_mask.any(dim=1).all():
            raise ValueError(
                "encoder_attention_mask must leave at least one VLM token valid "
                "for every batch item"
            )
        # PyTorch TransformerDecoder uses True for positions that must be ignored.
        return ~valid_mask

    def _append_state_memory(
        self,
        memory: torch.Tensor,
        state: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if state is None:
            return memory, memory_padding_mask
        if self.state_projection is None:
            raise ValueError("state was provided, but configured state_dim is 0")
        if not torch.is_tensor(state):
            raise TypeError(f"state must be a torch.Tensor or None, got {type(state)}")
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                f"state must have shape [B, N, D] or [B, D], got {tuple(state.shape)}"
            )
        if state.shape[0] != memory.shape[0]:
            raise ValueError(
                f"state batch size {state.shape[0]} does not match vl_embs "
                f"batch size {memory.shape[0]}"
            )
        if state.shape[1] <= 0:
            raise ValueError("state must contain at least one state token")
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state width must be configured state_dim={self.state_dim}, "
                f"got {state.shape[-1]}"
            )
        if not state.is_floating_point():
            raise TypeError(f"state must be floating point, got {state.dtype}")
        if state.device != memory.device:
            raise ValueError(
                f"state is on {state.device}, but vl_embs is on {memory.device}"
            )
        state_memory = self.state_projection(
            state.to(dtype=self.state_projection.weight.dtype)
        ).to(dtype=memory.dtype)
        memory = torch.cat((memory, state_memory), dim=1)
        if memory_padding_mask is not None:
            state_is_not_padding = torch.zeros(
                state.shape[:2], dtype=torch.bool, device=memory.device
            )
            memory_padding_mask = torch.cat(
                (memory_padding_mask, state_is_not_padding), dim=1
            )
        return memory, memory_padding_mask

    def forward_logits(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-axis logits with shape ``[B, 64, 3, 3]``."""

        self._validate_vl_embeddings(vl_embs)
        batch_size, memory_length, _ = vl_embs.shape
        memory = self.memory_projection(
            vl_embs.to(dtype=self.memory_projection.weight.dtype)
        )
        memory_padding_mask = self._memory_padding_mask(
            encoder_attention_mask,
            batch_size,
            memory_length,
            vl_embs.device,
        )
        memory, memory_padding_mask = self._append_state_memory(
            memory, state, memory_padding_mask
        )

        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.classifier(decoded).reshape(
            batch_size,
            self.action_horizon,
            self.action_dim,
            self.NUM_CLASSES,
        )
        if logits.shape != (
            batch_size,
            self.ACTION_HORIZON,
            self.NUM_AXES,
            self.NUM_CLASSES,
        ):
            raise RuntimeError(f"Unexpected discrete action logits shape: {tuple(logits.shape)}")
        return logits

    def _validate_action_mask(
        self, action_mask: torch.Tensor, expected_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        if not torch.is_tensor(action_mask):
            raise TypeError(
                f"action_mask must be a torch.Tensor, got {type(action_mask)}"
            )
        if tuple(action_mask.shape) != expected_shape:
            raise ValueError(
                f"action_mask must have shape {expected_shape}, "
                f"got {tuple(action_mask.shape)}"
            )
        valid_mask = self._binary_mask(action_mask, "action_mask")
        if not valid_mask.any():
            raise ValueError("action_mask must select at least one target")
        return valid_mask

    def _targets_to_class_indices(
        self, actions: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.action_target_format == "values":
            if not actions.is_floating_point() and actions.dtype == torch.bool:
                raise TypeError("value-formatted actions cannot have boolean dtype")
            values = self.action_class_values.to(
                device=actions.device, dtype=actions.dtype
            )
            if _unique_for_validation(values).numel() != self.NUM_CLASSES:
                raise ValueError(
                    f"action dtype {actions.dtype} cannot represent all configured "
                    "action_class_values distinctly"
                )
            matches = actions.unsqueeze(-1) == values
            match_count = matches.sum(dim=-1)
            invalid = valid_mask & (match_count != 1)
            if invalid.any():
                bad_values = _unique_for_validation(actions[invalid]).detach().cpu().tolist()
                raise ValueError(
                    "Every valid value-formatted action target must exactly equal one "
                    "configured action_class_values entry; invalid values include "
                    f"{bad_values[:8]}"
                )
            targets = matches.to(dtype=torch.long).argmax(dim=-1)
        else:
            if actions.dtype == torch.bool:
                raise TypeError("class-index action targets cannot have boolean dtype")
            if actions.is_floating_point():
                valid_actions = actions[valid_mask]
                if not torch.isfinite(valid_actions).all():
                    raise ValueError("Valid class-index targets must be finite")
                if not torch.equal(valid_actions, valid_actions.round()):
                    raise ValueError("Valid class-index targets must be exact integers")
            targets = actions.to(dtype=torch.long)
            invalid = valid_mask & ((targets < 0) | (targets >= self.NUM_CLASSES))
            if invalid.any():
                bad_values = _unique_for_validation(actions[invalid]).detach().cpu().tolist()
                raise ValueError(
                    f"Valid class-index targets must be in [0, {self.NUM_CLASSES - 1}]; "
                    f"invalid values include {bad_values[:8]}"
                )

        # Masked positions do not contribute to loss.  Give them a safe gather
        # index so arbitrary padding values cannot cause an indexing failure.
        return targets.masked_fill(~valid_mask, 0)

    def encode_targets(
        self,
        actions: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode configured target representation as per-axis class ids.

        Unlike training ``forward``, this utility accepts any leading shape as
        long as the final dimension contains the three axes.  When no mask is
        provided every target is validated.
        """

        if not torch.is_tensor(actions):
            raise TypeError(f"actions must be a torch.Tensor, got {type(actions)}")
        if actions.ndim < 1 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must end in action_dim={self.action_dim}, "
                f"got shape {tuple(actions.shape)}"
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(actions, dtype=torch.bool)
        else:
            if not torch.is_tensor(valid_mask):
                raise TypeError(
                    f"valid_mask must be a torch.Tensor, got {type(valid_mask)}"
                )
            if valid_mask.shape != actions.shape:
                raise ValueError(
                    "valid_mask must have the same shape as actions, got "
                    f"{tuple(valid_mask.shape)} and {tuple(actions.shape)}"
                )
            if valid_mask.device != actions.device:
                raise ValueError(
                    f"valid_mask is on {valid_mask.device}, but actions is on "
                    f"{actions.device}"
                )
            valid_mask = self._binary_mask(valid_mask, "valid_mask")
        return self._targets_to_class_indices(actions, valid_mask)

    def targets_to_values(
        self,
        actions: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return semantic values for value- or class-index-formatted targets.

        When a validity mask is supplied, arbitrary padding values are ignored
        during validation and decoded padding positions are set to the neutral
        command ``0`` before the trajectory is passed to the world model.
        """

        return decode_discrete_action_targets(
            actions,
            action_class_values=self.action_class_values,
            action_target_format=self.action_target_format,
            valid_mask=valid_mask,
        )

    def _weighted_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_target_loss = F.cross_entropy(
            logits.float().reshape(-1, self.NUM_CLASSES),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)

        weights = self.action_class_weights.to(
            device=logits.device, dtype=per_target_loss.dtype
        )
        expanded_weights = weights.view(
            1, 1, self.action_dim, self.NUM_CLASSES
        ).expand(logits.shape[0], logits.shape[1], -1, -1)
        selected_weights = expanded_weights.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        effective_weights = selected_weights * valid_mask.to(
            dtype=selected_weights.dtype
        )
        denominator = effective_weights.sum()
        if denominator.detach().item() <= 0:
            raise ValueError(
                "Valid targets have zero total action_class_weights; loss is undefined"
            )
        return (per_target_loss * effective_weights).sum() / denominator

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mask-aware, class-weighted per-axis cross entropy."""

        self._validate_vl_embeddings(vl_embs)
        if not torch.is_tensor(actions):
            raise TypeError(f"actions must be a torch.Tensor, got {type(actions)}")
        expected_shape = (vl_embs.shape[0], self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {tuple(actions.shape)}"
            )
        if actions.device != vl_embs.device:
            raise ValueError(
                f"actions is on {actions.device}, but vl_embs is on {vl_embs.device}"
            )
        if not torch.is_tensor(action_mask):
            raise TypeError(
                f"action_mask must be a torch.Tensor, got {type(action_mask)}"
            )
        if action_mask.device != vl_embs.device:
            raise ValueError(
                f"action_mask is on {action_mask.device}, but vl_embs is on {vl_embs.device}"
            )

        valid_mask = self._validate_action_mask(action_mask, expected_shape)
        targets = self._targets_to_class_indices(actions, valid_mask)
        logits = self.forward_logits(
            vl_embs,
            state=state,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self._weighted_cross_entropy(logits, targets, valid_mask)

    def decode_class_indices(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Decode class ids using the checkpoint-persisted class/value mapping."""

        if not torch.is_tensor(class_indices):
            raise TypeError("class_indices must be a torch.Tensor")
        if class_indices.dtype == torch.bool:
            raise TypeError("class_indices cannot have boolean dtype")
        if class_indices.is_floating_point():
            if not torch.isfinite(class_indices).all():
                raise ValueError("class_indices must be finite")
            if not torch.equal(class_indices, class_indices.round()):
                raise ValueError("class_indices must contain exact integers")
        indices = class_indices.to(dtype=torch.long)
        if ((indices < 0) | (indices >= self.NUM_CLASSES)).any():
            raise ValueError(
                f"class_indices must be in [0, {self.NUM_CLASSES - 1}]"
            )
        values = self.action_class_values.to(device=indices.device)
        return values[indices]

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        num_candidates: int = 1,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Decode one argmax plan or ``K`` independently sampled plans.

        Returns ``[B, 64, 3]`` when ``num_candidates == 1`` and
        ``[B, K, 64, 3]`` otherwise.  Returned tensors contain semantic action
        values, never internal class ids.
        """

        if isinstance(num_candidates, bool):
            raise ValueError("num_candidates must be a positive integer")
        try:
            parsed_num_candidates = int(num_candidates)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_candidates must be a positive integer") from exc
        if parsed_num_candidates != num_candidates:
            raise ValueError("num_candidates must be a positive integer")
        num_candidates = parsed_num_candidates
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")

        logits = self.forward_logits(
            vl_embs,
            state=state,
            encoder_attention_mask=encoder_attention_mask,
        )
        if num_candidates == 1:
            class_indices = logits.argmax(dim=-1)
            return self.decode_class_indices(class_indices)

        probabilities = logits.float().softmax(dim=-1)
        flat_probabilities = probabilities.reshape(-1, self.NUM_CLASSES)
        sampled = torch.multinomial(
            flat_probabilities,
            num_samples=num_candidates,
            replacement=True,
            generator=generator,
        )
        sampled = sampled.reshape(
            logits.shape[0],
            self.action_horizon,
            self.action_dim,
            num_candidates,
        ).permute(0, 3, 1, 2)
        return self.decode_class_indices(sampled)

    @property
    def device(self) -> torch.device:
        return self.action_queries.device

    @property
    def dtype(self) -> torch.dtype:
        return self.action_queries.dtype


def get_discrete_action_model(config: Any) -> DiscreteActionHead:
    """Build a :class:`DiscreteActionHead` from a global or action config."""

    return DiscreteActionHead(config)
