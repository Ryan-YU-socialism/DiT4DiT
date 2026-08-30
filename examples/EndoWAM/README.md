# EndoWAM `endowam_pseudo_z60` operations

本目录保存 EndoWAM 数据同步、环境准备、训练 preflight、DeepSpeed launch 和断线恢复脚本。代码结构与数据契约见 [`docs/endowam_pseudo_z60.md`](../../docs/endowam_pseudo_z60.md)。

> 旧 2-card 正式 run 在首个周期 checkpoint 之前结束，没有可恢复 checkpoint。本页命令是运维手册；当前是否在训练必须以 ren5 的 tmux、worker、GPU 和 completion manifest 为准。

## Recommended ren5 recipe

ren5 当前配方使用三张 RTX 3090 24GB（CUDA indices 0、1、2）。GPU 2 曾有 NVML/PCIe 故障，因此每次正式启动前都必须重新通过 CUDA allocation 和 3-rank NCCL collective 检查。

| 项目 | 值 |
| --- | --- |
| GPU | 3 × RTX 3090 24GB |
| PyTorch / CUDA | 2.5.1 / 12.4 |
| precision | BF16 |
| DeepSpeed | ZeRO-2 + CPU optimizer offload |
| batch | per-device 3，accumulation 1，global 9 |
| CPU threads | OpenMP 6/rank；MKL/OpenBLAS/NumExpr 1 |
| max steps | 100,000 |
| warmup | 500 |
| save / eval / log | 5,000 / 100 / 10 steps |
| prompt length | 64 tokens |
| checkpoint | frozen text encoder/VAE weights excluded |

默认远端路径：

```text
repository  /mnt/data-hdd2/ljs/DiT4DiT
conda env   /mnt/data-hdd2/ljs/.conda/envs/endoguard
dataset     /mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60
base model  /mnt/data-hdd2/ljs/models/Cosmos-Predict2.5-2B
run root    /mnt/data-hdd3/ljs/experiments/DiT4DiT
```

这些路径是 ren5 的部署默认值。脚本都允许用同名环境变量覆盖，不应为另一台主机直接硬改公共脚本。

## Files

| 文件 | 用途 |
| --- | --- |
| `sync_endowam_ren5.sh` | ren5 上直接从 Google Drive 同步，直连/Clash 交替重试 |
| `sync_endowam_from_drive.sh` | 通用 authenticated-rclone 同步脚本 |
| `download_cosmos_ren5.sh` | 下载并核验本地 Cosmos base model |
| `setup_ren5_env.sh` | 创建隔离 conda prefix，安装 PyTorch 2.5.1+cu124 和项目依赖 |
| `build_ren5_nvml_filter.sh` | 构建仅对训练进程生效的 NVML device-count filter |
| `run_endowam_ren5_2x3090.sh` | ren5 环境/硬件默认值包装器 |
| `run_endowam_ren5_3x3090.sh` | ren5 三卡正式配方包装器 |
| `run_endowam_4xh800.sh` | 共用 preflight 和 Accelerate launch 实现，名字是历史遗留 |
| `supervise_endowam_ren5.sh` | 防重复、锁参数、auto-resume、有限重试 |
| `supervise_endowam_ren5_3x3090.sh` | 三卡 supervisor 入口；复用通用守护逻辑 |
| `run_endowam_4xpro6000.sh` | 4 × RTX PRO 6000 替代配方 |
| `run_endowam_2xpro6000.sh` | 2 × RTX PRO 6000 替代配方 |

## One-time preparation on ren5

以下操作应在 ren5 仓库根目录执行。

### 1. Sync the dataset

先在远端用户环境配置 `rclone` Google Drive 登录。凭据、账号和 token 不得写入 Git。ren5 的可重试同步：

```bash
bash examples/EndoWAM/train_files/sync_endowam_ren5.sh
```

通用主机可显式提供 remote path 和目标位置：

```bash
bash examples/EndoWAM/train_files/sync_endowam_from_drive.sh \
  gdrive:endowam_pseudo_z60 \
  /path/to/datasets/endowam_pseudo_z60
```

脚本必须核验 `ureter`、`ercp`、`esophagus` 三个子集。不要通过本地 Mac 中转数据。

### 2. Download the base model

```bash
bash examples/EndoWAM/train_files/download_cosmos_ren5.sh
```

配置固定使用 Cosmos-Predict2.5-2B 的 `diffusers/base/post-trained` revision，并在训练时启用 `local_files_only`。若移动目录，通过 `BASE_MODEL` 覆盖。

### 3. Create the isolated environment

```bash
bash examples/EndoWAM/train_files/setup_ren5_env.sh
```

该脚本创建基础 `dit4dit-ren5` conda prefix，隔离共享 profile 中的 user-site packages 和 pip extra index。当前三卡运行使用其经过验证的独立 clone `endoguard`；必须核对 PyTorch 2.5.1+cu124、Accelerate 1.12.0 和 DeepSpeed 0.18.4。

## Preflight before every run

正式运行前手工确认：

```bash
cd /mnt/data-hdd2/ljs/DiT4DiT
git status --short --branch
git rev-parse HEAD

CUDA_VISIBLE_DEVICES=0,1,2 \
  /mnt/data-hdd2/ljs/.conda/envs/endoguard/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
PY

du -sh /mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60
df -h /mnt/data-hdd3/ljs/experiments/DiT4DiT
pgrep -af 'DiT4DiT/training/train.py|supervise_endowam_ren5' || true
tmux list-sessions || true
```

启动器还会自动检查：工作树是否 clean、三个数据子集、base model 目录、GPU 数量/型号、BF16、Git commit、输出目录、整数参数、resume 模式和磁盘状态。

单个周期 checkpoint 实测约 28 GB。100,000 steps、每 5,000 steps 保存会产生 20 个周期 checkpoint，约 560 GB；加上 final model、日志、缓存和安全余量，建议 run volume 至少留 750 GB。当前没有自动 checkpoint rotation。

## Smoke test

任何代码、环境或关键配置变更后，使用新的 `RUN_ID` 做短测试。不要复用正式 run ID：

```bash
RUN_ID=dit4dit_endowam_smoke_$(date +%Y%m%d_%H%M%S) \
MAX_TRAIN_STEPS=20 \
NUM_WARMUP_STEPS=2 \
SAVE_INTERVAL=20 \
EVAL_INTERVAL=20 \
bash examples/EndoWAM/train_files/run_endowam_ren5_3x3090.sh
```

至少确认：两个 rank 正常、loss finite、峰值显存不过载、step time 稳定、checkpoint 发布 manifest，并用同一个 smoke `RUN_ID` 验证一次 `RESUME=auto`。

## Long run with tmux and supervisor

长训练使用 supervisor，而不是直接 runner。supervisor 会锁定 commit 和训练语义参数、防止重复启动、最多重试 3 次，并为每次 attempt 保存日志。

在仓库根目录启动 detached tmux：

```bash
tmux new-session -d -s endowam-ren5 \
  'cd /mnt/data-hdd2/ljs/DiT4DiT && exec bash examples/EndoWAM/train_files/supervise_endowam_ren5_3x3090.sh'
```

查看而不进入：

```bash
tmux capture-pane -pt endowam-ren5:0 -S -120
tail -n 120 \
  /mnt/data-hdd3/ljs/experiments/DiT4DiT/dit4dit_endowam_pseudo_z60_ren5_3x3090/supervisor_logs/supervisor.log
nvidia-smi
```

进入/离开：

```bash
tmux attach -t endowam-ren5
# detach: Ctrl-b, then d
```

网络断开不会结束 tmux 中的训练。训练依赖都来自本地 dataset/base model/cache，因此正常训练 step 不依赖 Google Drive 或外网；W&B 默认 offline。

## Stop safely

优先在 tmux 内向 supervisor 发送一次 `Ctrl-C`，等待所有 rank 退出：

```bash
tmux send-keys -t endowam-ren5:0 C-c
```

然后确认没有残留 worker：

```bash
pgrep -af 'DiT4DiT/training/train.py|supervise_endowam_ren5' || true
nvidia-smi
```

确认 session 已经不再承担任务后再关闭：

```bash
tmux kill-session -t endowam-ren5
```

人工中断不会被 supervisor 自动重试。由于 pipeline 使用 `tee`，中断时日志可能额外显示 `Supervisor logging failed rc=130`；仍应以上面的进程/GPU 检查为准。不要用粗范围 `pkill` 影响其他用户任务。

## Resume behavior

runner 和 supervisor 默认 `RESUME=auto`。只有以下两项同时存在且 manifest 合法时才会恢复：

```text
checkpoints/steps_N_state/
checkpoints/steps_N_complete.json
```

恢复包括 model、optimizer、scheduler、RNG 和 dataloader position。配置设置 `checkpoint_exclude_frozen_parameters=true`，因此冻结的 Cosmos text encoder/VAE 会从同一个本地 base model 重载；base model 不可丢失或静默替换。

首个 checkpoint 是 step 5,000。在这之前停止，没有 full-state 可恢复内容，下次同一 run 会从 step 0 开始。不要把孤立的 `.incomplete` 目录或单个 `.pt` 当成有效 checkpoint。

supervisor 会把参数锁在：

```text
<run_dir>/supervisor_logs/launch_config.env
```

重新启动同一 `RUN_ID` 时参数必须完全一致。要改变 batch、schedule、数据、模型或配置，使用新的 `RUN_ID`；不要删除 lock 文件来伪装兼容恢复。

## Outputs and logs

```text
<run_root>/<run_id>/
├── config.yaml
├── dataset_statistics.json
├── summary.jsonl
├── train_YYYYMMDD_HHMMSS.log
├── supervisor_logs/
│   ├── supervisor.lock
│   ├── launch_config.env
│   ├── attempts.tsv
│   ├── supervisor.log
│   └── attempt_N_YYYYMMDDTHHMMSS.log
├── checkpoints/
│   ├── steps_N_state/
│   └── steps_N_complete.json
└── final_model/
```

默认 `save_consolidated_checkpoints=false`，所以周期保存以 resumable state 为主，不保证存在 `steps_N_pytorch_model.pt`。不要把 checkpoint、日志、W&B 数据或模型权重提交进 Git。

## Alternative AutoDL recipes

ren5 不可用时，优先单机 4 × RTX PRO 6000 96GB；对应 launcher 为 `run_endowam_4xpro6000.sh`，BF16 + ZeRO-2，默认 global batch 32。2 卡替代方案用 `run_endowam_2xpro6000.sh`，通过更高 gradient accumulation 保持 global batch 32。

RTX PRO 6000 是 Blackwell 配方，需使用兼容的 CUDA 12.8 PyTorch build 和 Triton ≥ 3.3。租机页面的 GPU 型号不足以确认通信拓扑；创建实例后仍要检查 `nvidia-smi topo -m`。AutoDL 路径默认在 `/root/autodl-tmp`，与 ren5 路径不同。

双 3090 配方仍作为 GPU 2 再次故障时的 fallback，global batch 为 6；不要直接用双卡 ZeRO-2 state 恢复三卡 run。H800/PRO 6000 配方仍保留其原始 80,000-step 默认值；用户指定的 100,000/500/5,000/100 schedule 已落实在 ren5 配置和 wrapper 中。不要混用硬件配方的 YAML、Accelerate YAML 和 DeepSpeed JSON。
