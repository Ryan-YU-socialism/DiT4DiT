# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modified by [Teli Ma/ HKUST GZ] in [2025]. 
# Modification: [modify more efficient distributed training and from pre-training mode to training mode].


# Standard Library
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from DiT4DiT.training.trainer_utils.trainer_tools import normalize_dotlist_args
from DiT4DiT.model.framework import build_framework
from DiT4DiT.training.trainer_utils.trainer_tools import TrainerUtils
from DiT4DiT.training.trainer_utils.trainer_tools import build_param_lr_groups
from DiT4DiT.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
# 获取本地 Rank（Ray 通常会自动设置 LOCAL_RANK 环境变量）
# local_rank = int(os.environ.get("LOCAL_RANK", 0))

# 强制绑定设备，消除警告
# torch.cuda.set_device(local_rank)


def build_accelerator(cfg) -> Accelerator:
    """Build Accelerate/DeepSpeed after the run config has been loaded.

    The previous module-level construction hard-coded the generic ZeRO-2 file
    and forced gradient accumulation to one.  That made a launcher's selected
    DeepSpeed config and ``trainer.gradient_accumulation_steps`` ineffective.
    Keeping the values synchronized here is important for reproducible global
    batch sizes on both single- and multi-GPU jobs.
    """

    default_ds_config = "DiT4DiT/config/deepseeds/ds_config.yaml"
    configured_ds_path = cfg.trainer.get("deepspeed_config", default_ds_config)
    ds_config_path = os.environ.get(
        "DIT4DIT_DEEPSPEED_CONFIG", str(configured_ds_path)
    )
    if not Path(ds_config_path).is_file():
        raise FileNotFoundError(
            f"DeepSpeed config does not exist: {ds_config_path}. "
            "Set trainer.deepspeed_config or DIT4DIT_DEEPSPEED_CONFIG."
        )

    accumulation_steps = int(cfg.trainer.gradient_accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("trainer.gradient_accumulation_steps must be positive")

    with open(ds_config_path, "r", encoding="utf-8") as config_file:
        ds_config = yaml.safe_load(config_file)
    ds_accumulation_steps = ds_config.get("gradient_accumulation_steps", "auto")
    if (
        ds_accumulation_steps != "auto"
        and int(ds_accumulation_steps) != accumulation_steps
    ):
        raise ValueError(
            "DeepSpeed gradient_accumulation_steps conflicts with the run config: "
            f"{ds_accumulation_steps} != {accumulation_steps}."
        )

    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_path)
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=accumulation_steps,
    )
    accelerator.print(accelerator.state)
    return accelerator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        raw_cfg = cfg._cfg if isinstance(cfg, AccessTrackedConfig) else cfg
        OmegaConf.save(config=raw_cfg, f=output_dir / "config.yaml")
        logger.info(f"Saved complete run configuration to {output_dir / 'config.yaml'}")

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from DiT4DiT.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    dataset_label = cfg.datasets.vla_data.get(
        "dataset_name",
        cfg.datasets.vla_data.get("data_mix", "<unconfigured>"),
    )
    logger.info(f"Creating VLA Dataset `{dataset_label}`")
    # Access in main process so this key is tracked and persisted by AccessTrackedConfig.
    action_video_freq_ratio = cfg.datasets.vla_data.get("action_video_freq_ratio", 1)
    logger.info(f"Using action_video_freq_ratio={action_video_freq_ratio}")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer(model, cfg) -> torch.optim.Optimizer:
    """Create the optimizer before DeepSpeed wraps the model."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    return optimizer


def setup_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a scheduler bound to the optimizer that performs updates.

    Accelerate replaces AdamW with DeepSpeedCPUAdam for the ren5 ZeRO-3
    recipe. Creating the scheduler before ``accelerator.prepare`` leaves it
    attached to the discarded AdamW instance and silently trains at LR=0.
    """
    scheduler_specific_kwargs = dict(
        cfg.trainer.scheduler_specific_kwargs.items()
    )
    if (
        cfg.trainer.lr_scheduler_type == "cosine_with_min_lr"
        and "min_lr" in scheduler_specific_kwargs
        and "min_lr_rate" not in scheduler_specific_kwargs
    ):
        # Transformers derives this ratio from optimizer.defaults["lr"], but
        # DeepSpeed's ZeRO optimizer intentionally exposes no defaults mapping.
        # Use the configured base LR explicitly; this is algebraically
        # equivalent and keeps the same ratio for every parameter group.
        base_learning_rate = float(cfg.trainer.learning_rate.base)
        if base_learning_rate <= 0:
            raise ValueError("trainer.learning_rate.base must be positive")
        scheduler_specific_kwargs["min_lr_rate"] = (
            float(scheduler_specific_kwargs.pop("min_lr"))
            / base_learning_rate
        )

    return get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = None
        self.accelerator = accelerator
        self.resume_state_checkpoint: Optional[str] = None

        # training status tracking
        self.completed_steps = 0
        self.consumed_data_batches = 0
        self.total_batch_size = self._calculate_total_batch_size()
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Guard: if everything is frozen, Deepspeed ZeRO optimizer init will crash with an opaque
        # `torch.cat(): expected a non-empty list of Tensors`.
        # any_trainable = any(p.requires_grad for p in self.model.parameters())
        # if not any_trainable:
        #     # Show a few top-level module names to help locate what got frozen.
        #     top_modules = list(dict(self.model.named_children()).keys())
        #     raise RuntimeError(
        #         "No trainable parameters found after freezing. "
        #         "Please check `trainer.freeze_modules` and any backbone wrappers that set requires_grad_(False).\n"
        #         f"- trainer.freeze_modules: {freeze_modules!r}\n"
        #         f"- top-level modules: {top_modules}\n"
        #         "Fix: ensure at least the action head stays trainable (e.g., do not freeze `action_model`)."
        #     )

        # # IMPORTANT: the optimizer was built before we freeze modules (see main()).
        # # Deepspeed ZeRO assumes optimizer param groups contain trainable params; if a group becomes empty after
        # # filtering `requires_grad`, it may crash during flattening.
        # if self.optimizer is not None and hasattr(self.optimizer, "param_groups"):
        #     for group in self.optimizer.param_groups:
        #         group["params"] = [p for p in group.get("params", []) if getattr(p, "requires_grad", False)]
        #     # drop empty groups
        #     self.optimizer.param_groups = [g for g in self.optimizer.param_groups if len(g.get("params", [])) > 0]
        #     total_opt_params = sum(len(g.get("params", [])) for g in self.optimizer.param_groups)
        #     if total_opt_params == 0:
        #         raise RuntimeError(
        #             "Optimizer has 0 trainable parameters after freezing/pruning. "
        #             "This will crash Deepspeed ZeRO.\n"
        #             f"- trainer.freeze_modules: {freeze_modules!r}\n"
        #             "Fix: ensure your action head parameters are included in the optimizer param_groups."
        #         )

        # Print parameter statistics after freezing
        if not dist.is_initialized() or dist.get_rank() == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            logger.info("=" * 80)
            logger.info("📊 Model Parameter Statistics (after freezing):")
            logger.info(f"  Total parameters:      {total_params:,} ({total_params / 10**6:.3f}M)")
            logger.info(f"  Trainable parameters:  {trainable_params:,} ({trainable_params / 10**6:.3f}M)")
            logger.info(f"  Frozen parameters:     {frozen_params:,} ({frozen_params / 10**6:.3f}M)")
            logger.info(f"  Trainable ratio:       {trainable_params / total_params * 100:.2f}%")
            logger.info("=" * 80)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        # DeepSpeed may replace AdamW with DeepSpeedCPUAdam. Build the raw
        # scheduler only now, bound directly to the optimizer that performs
        # updates. AcceleratedScheduler would over-step it by world size when
        # split_batches=False, so register it as an ordinary checkpointable.
        actual_optimizer = self._get_actual_optimizer()
        model_optimizer = getattr(self.model, "optimizer", None)
        if model_optimizer is not None and model_optimizer is not actual_optimizer:
            raise RuntimeError(
                "Accelerate and DeepSpeed expose different optimizer instances."
            )
        # The Accelerate wrapper is a torch Optimizer (required by PyTorch's
        # LRScheduler) and its param_groups transparently reference ZeRO-3's
        # underlying optimizer. The raw ZeRO optimizer is not a torch Optimizer.
        self.lr_scheduler = setup_lr_scheduler(self.optimizer, self.config)
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        if self.resume_state_checkpoint:
            self._load_checkpoint(self.resume_state_checkpoint)
        self._sync_scheduler_lrs()

        if self.accelerator.is_main_process:
            logger.info(
                "Prepared optimizer learning rates: %s",
                [group["lr"] for group in self.optimizer.param_groups],
            )

        self._init_wandb()


    def _get_actual_optimizer(self):
        """Return the optimizer owned by the DeepSpeed engine, if wrapped."""
        return getattr(self.optimizer, "optimizer", self.optimizer)

    def _sync_scheduler_lrs(self):
        """Validate and publish scheduled LRs to the real optimizer groups."""
        scheduled_lrs = [float(lr) for lr in self.lr_scheduler.get_last_lr()]
        if len(scheduled_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError(
                "Scheduler/optimizer parameter-group mismatch: "
                f"{len(scheduled_lrs)} != {len(self.optimizer.param_groups)}"
            )
        if not all(np.isfinite(lr) and lr >= 0 for lr in scheduled_lrs):
            raise RuntimeError(f"Scheduler produced invalid learning rates: {scheduled_lrs}")
        for group, learning_rate in zip(
            self.optimizer.param_groups, scheduled_lrs
        ):
            group["lr"] = learning_rate

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Resolve a full-state resume or an explicit weights-only warm start."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.resume_state_checkpoint = resume_from_checkpoint
                logger.info(
                    f"Will resume full training state from {resume_from_checkpoint}, "
                    f"steps: {self.completed_steps}"
                )
                return None
            logger.warning(
                f"No complete state checkpoint found in {self.checkpoint_dir}. "
                "Starting training from scratch."
            )
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, scheduler, and RNG state."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    @staticmethod
    def _atomic_torch_save(value, destination: str) -> None:
        """Write a checkpoint without exposing a partially written final file."""
        temporary = f"{destination}.incomplete"
        with open(temporary, "wb") as output_file:
            torch.save(value, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)

    @staticmethod
    def _atomic_json_save(value: dict, destination: str) -> None:
        temporary = f"{destination}.incomplete"
        with open(temporary, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)

    def _save_checkpoint(self):
        """Atomically publish resumable state and optional inference weights."""
        checkpoint_stem = f"steps_{self.completed_steps}"
        state_checkpoint = os.path.join(
            self.checkpoint_dir, f"{checkpoint_stem}_state"
        )
        temporary_state_checkpoint = f"{state_checkpoint}.incomplete"
        model_checkpoint = os.path.join(
            self.checkpoint_dir, f"{checkpoint_stem}_pytorch_model.pt"
        )
        manifest_path = os.path.join(
            self.checkpoint_dir, f"{checkpoint_stem}_complete.json"
        )

        # Every rank performs the same conflict check, avoiding a rank-zero
        # exception that would strand peers at the next collective barrier.
        if os.path.exists(manifest_path):
            raise FileExistsError(
                f"Refusing to overwrite complete checkpoint: {manifest_path}"
            )
        if self.accelerator.is_main_process:
            for stale_path in (temporary_state_checkpoint, state_checkpoint):
                if os.path.exists(stale_path):
                    shutil.rmtree(stale_path)
        self.accelerator.wait_for_everyone()

        # For DeepSpeed this collectively saves ZeRO-3 model shards, CPUAdam,
        # RNG state, and the separately registered raw LR scheduler.
        self.accelerator.save_state(temporary_state_checkpoint)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.replace(temporary_state_checkpoint, state_checkpoint)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

            # Publish the resumable state before optional consolidation. If a
            # later gather/export fails, auto-resume can still recover safely.
            self._atomic_json_save(
                {
                    "format_version": 1,
                    "steps": self.completed_steps,
                    "consumed_data_batches": self.consumed_data_batches,
                    "state_dir": os.path.basename(state_checkpoint),
                    "model_file": None,
                },
                manifest_path,
            )
            self.accelerator.print(f"✅ Checkpoint saved at {state_checkpoint}")
        self.accelerator.wait_for_everyone()

        save_consolidated = bool(
            self.config.trainer.get("save_consolidated_checkpoints", False)
        )
        if save_consolidated:
            # ZeRO-3 consolidation is collective. This export is not required
            # for resume and can be disabled on large models.
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                self._atomic_torch_save(state_dict, model_checkpoint)
                self._atomic_json_save(
                    {
                        "format_version": 1,
                        "steps": self.completed_steps,
                        "consumed_data_batches": self.consumed_data_batches,
                        "state_dir": os.path.basename(state_checkpoint),
                        "model_file": os.path.basename(model_checkpoint),
                    },
                    manifest_path,
                )
            del state_dict
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # Log the learning rates used by the real DeepSpeed optimizer.
                for index, group in enumerate(
                    self.optimizer.param_groups
                ):
                    group_name = group.get("name", f"group_{index}")
                    metrics[f"learning_rate/{group_name}"] = float(group["lr"])

                # add epoch info
                metrics["epoch"] = round(self.completed_steps * self.config.trainer.gradient_accumulation_steps  / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"[Exp: {self.config.run_id}] Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_epoch_count = 0
        consumed_batches = self.consumed_data_batches
        if self.resume_state_checkpoint and consumed_batches > 0:
            batches_per_epoch = len(self.vla_train_dataloader)
            if batches_per_epoch <= 0:
                raise RuntimeError("Prepared training dataloader is empty")
            epoch, batch_offset = divmod(consumed_batches, batches_per_epoch)
            self.vla_epoch_count = epoch
            dataset = getattr(self.vla_train_dataloader, "dataset", None)
            set_dataset_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_dataset_epoch):
                set_dataset_epoch(epoch)
            sampler = getattr(self.vla_train_dataloader, "sampler", None)
            set_sampler_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_sampler_epoch):
                set_sampler_epoch(epoch)
            resume_dataloader = self.accelerator.skip_first_batches(
                self.vla_train_dataloader, num_batches=batch_offset
            )
            self.vla_iter = iter(resume_dataloader)
            logger.info(
                "Resuming data stream at epoch=%s, batch_offset=%s",
                epoch,
                batch_offset,
            )
        else:
            self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        self.consumed_data_batches += 1
        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            did_step = bool(self.accelerator.sync_gradients)
            if did_step:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # Only eval/log/save on real optimizer steps (end of accumulation window)
            if did_step:
                # evaluate model (skip action eval in video-only mode)
                video_fm_only = getattr(self.model, "video_fm_only", False)
                if (not video_fm_only) and (self.completed_steps % self.config.trainer.eval_interval == 0):
                    step_metrics = self.eval_action_model(step_metrics)

                # record metrics
                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                # save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        action_mask = [example["action_mask"] for example in examples] # [B, len, action_dim]
        eval_with_stage1 = bool(
            self.config.trainer.get("eval_with_stage1", True)
        )
        output_dict = self.model.predict_action(
            examples=examples,
            use_ddim=True,
            num_ddim_steps=20,
            disable_stage1=not eval_with_stage1,
        )

        if self.accelerator.is_main_process:
            predicted_actions = output_dict["normalized_actions"]  # B, T, D
            action_horizon = int(self.model.config.framework.action_model.future_action_window_size) + 1
            action_dim = int(self.model.config.framework.action_model.action_dim)
            actions = np.asarray(actions)[:, -action_horizon:, :action_dim]
            action_mask = np.asarray(action_mask)[:, -action_horizon:, :action_dim].astype(bool)

            if bool(getattr(self.model, "discrete_actions", False)):
                # Convert class-index datasets through the checkpoint's explicit
                # mapping before comparing with decoded policy commands.
                target_values = self.model.action_model.targets_to_values(
                    torch.as_tensor(actions, device=self.model.device),
                    valid_mask=torch.as_tensor(action_mask, device=self.model.device),
                ).detach().cpu().numpy()
                correct = predicted_actions == target_values
                valid_axes = int(action_mask.sum())
                step_metrics["action_axis_accuracy"] = (
                    float(correct[action_mask].mean()) if valid_axes else 0.0
                )

                valid_steps = action_mask.any(axis=-1)
                step_correct = np.logical_or(correct, ~action_mask).all(axis=-1)
                step_metrics["action_step_accuracy"] = (
                    float(step_correct[valid_steps].mean()) if valid_steps.any() else 0.0
                )
            else:
                # Apply action_mask: only compute MSE on valid (True) dimensions.
                masked_diff = (predicted_actions - actions) * action_mask
                denominator = action_mask.sum()
                step_metrics["mse_score"] = (
                    float((masked_diff ** 2).sum() / denominator) if denominator else 0.0
                )

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  len(vla_train_dataloader) = {len(self.vla_train_dataloader)}")
            logger.info(f"  len(vla_train_dataloader.dataset) = {len(self.vla_train_dataloader.dataset)}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            # NOTE: In some DeepSpeed/Accelerate configurations, autograd can be globally disabled in engine.forward().
            # We force-enable grad here to ensure `loss.grad_fn` exists in training.

            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)

                    action_loss = output_dict.get("action_loss", None)
                    future_video_loss = output_dict.get("future_video_loss", None)

                    # Validate at least one loss exists
                    if action_loss is None and future_video_loss is None:
                        raise KeyError(
                            "Model output must contain either `action_loss` or `future_video_loss` for training."
                        )
                    if action_loss is not None and not torch.is_tensor(action_loss):
                        raise TypeError(
                            f"`action_loss` must be a torch.Tensor, got {type(action_loss)}. "
                            "This usually means model.forward returned a Python float or did `.item()`."
                        )
                    if future_video_loss is not None and not torch.is_tensor(future_video_loss):
                        raise TypeError(
                            f"`future_video_loss` must be a torch.Tensor, got {type(future_video_loss)}."
                        )

                    # Compute total loss based on training mode:
                    #   video:  total = future_video_loss * scale
                    #   action: total = action_loss
                    #   joint:  total = action_loss + future_video_loss * scale
                    future_video_loss_scaled = None
                    video_loss_scale = 1.0
                    if future_video_loss is not None:
                        try:
                            video_loss_scale = float(getattr(getattr(self.config.trainer, "loss_scale", None), "future_video", 1.0))
                        except Exception:
                            video_loss_scale = 1.0
                        future_video_loss_scaled = future_video_loss * video_loss_scale

                    if action_loss is not None and future_video_loss is not None:
                        # joint: action + video auxiliary
                        total_loss = action_loss + future_video_loss_scaled
                    elif action_loss is not None:
                        # action only
                        total_loss = action_loss
                    else:
                        # video only
                        total_loss = future_video_loss_scaled

                    # VLA backward propagation (keep inside enable_grad scope)
                    self.accelerator.backward(total_loss)

            # Only step optimizer / scheduler when gradients are synchronized (i.e., end of accumulation window).
            if self.accelerator.sync_gradients:
                # gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self._sync_scheduler_lrs()
                self.optimizer.zero_grad(set_to_none=True)

        step_metrics = {}
        if action_loss is not None:
            step_metrics["action_dit_loss"] = action_loss.item()
        if future_video_loss is not None:
            # raw auxiliary loss
            step_metrics["future_video_loss"] = future_video_loss.item() if torch.is_tensor(future_video_loss) else float(future_video_loss)
        if future_video_loss_scaled is not None:
            step_metrics["future_video_loss_scaled"] = (
                future_video_loss_scaled.item()
                if torch.is_tensor(future_video_loss_scaled)
                else float(future_video_loss_scaled)
            )
        # total loss (for quick monitoring only)
        step_metrics["total_loss"] = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
        if torch.cuda.is_available():
            gibibyte = float(1024 ** 3)
            step_metrics["gpu_memory_allocated_gib"] = (
                torch.cuda.memory_allocated(self.accelerator.device) / gibibyte
            )
            step_metrics["gpu_memory_reserved_gib"] = (
                torch.cuda.memory_reserved(self.accelerator.device) / gibibyte
            )
            step_metrics["gpu_memory_peak_gib"] = (
                torch.cuda.max_memory_allocated(self.accelerator.device) / gibibyte
            )
        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        save_final_training_state = bool(
            self.config.trainer.get("save_final_training_state", True)
        )
        final_manifest = os.path.join(
            self.checkpoint_dir, f"steps_{self.completed_steps}_complete.json"
        )
        should_save_training_state = (
            save_final_training_state and not os.path.exists(final_manifest)
        )
        if dist.is_initialized():
            decision = torch.tensor(
                [int(should_save_training_state if dist.get_rank() == 0 else False)],
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            dist.broadcast(decision, src=0)
            should_save_training_state = bool(decision.item())
        if should_save_training_state:
            self._save_checkpoint()

        save_final_model = bool(self.config.trainer.get("save_final_model", True))
        if save_final_model:
            # ZeRO-3 requires every rank to participate in consolidation.
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                final_checkpoint = os.path.join(self.config.output_dir, "final_model")
                os.makedirs(final_checkpoint, exist_ok=True)
                self._atomic_torch_save(
                    state_dict, os.path.join(final_checkpoint, "pytorch_model.pt")
                )
                logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            del state_dict
        else:
            logger.info("Training complete. Final model save disabled for this run.")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    accelerator = build_accelerator(cfg)

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # The scheduler is created only after DeepSpeed prepares the optimizer.
    optimizer = setup_optimizer(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="xxx.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
