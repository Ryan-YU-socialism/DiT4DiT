# EndoWAM `endowam_pseudo_z60` developer guide

本文是 `linjs` 分支上 EndoWAM 适配的代码交接文档。目标是让新的开发者能够先理解数据、模型、配置和恢复边界，再安全修改代码。面向 ren5 的日常运行命令见 [`examples/EndoWAM/README.md`](../examples/EndoWAM/README.md)。

## Handoff snapshot

| 项目 | 当前值 |
| --- | --- |
| 文档基线 | `linjs@c77f7c0` |
| 数据集 | `endowam_pseudo_z60`，LeRobot v2.x |
| 子集 | `ureter`、`ercp`、`esophagus`，等权 mixture |
| 动作 | 64 steps × 3 axes，语义值 `{-1, 0, +1}` |
| 状态 | 1 step × 3 axes，同样是离散语义值 |
| 图像 | `video.endoscope`，抽取 `[0, 16, 32, 48, 63]` |
| 主配置 | `DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_ren5_2x3090.yaml` |
| ren5 配方 | 2 × RTX 3090，BF16，ZeRO-2 + CPU optimizer offload |
| 全局 batch | `3 × 2 × 1 = 6` |
| 训练计划 | 100,000 steps；warmup 500；save 5,000；eval 100 |
| 当前运行状态 | 已停止；此前正式 run 在首个 checkpoint 前停止，没有可恢复 state |

最后一行是交接时的运行状态，不应被理解为仓库会自动启动或保持训练。修改前应重新检查远端进程、GPU 和 run 目录。

## Start reading here

建议按以下顺序阅读，先掌握配置和数据契约，再进入模型实现：

1. `DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_ren5_2x3090.yaml`：完整实验语义。
2. `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`：三个子集的注册和采样权重。
3. `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`：EndoWAM 字段映射、64-step action window 和禁止归一化的 transform。
4. `DiT4DiT/dataloader/gr00t_lerobot/datasets.py`：LeRobot sample、padding、mask 和视频 delta 的实际装配。
5. `DiT4DiT/model/framework/DiT4DiT.py`：joint forward、动作监督截取和离散分支路由。
6. `DiT4DiT/model/modules/action_model/discrete_action_head.py`：三轴三分类 action head。
7. `DiT4DiT/model/modules/vlm/Cosmos25.py`：Cosmos prompt/video encoding、动作条件 token 和未来视频 loss。
8. `DiT4DiT/model/framework/stage1.py`：多候选 rollout、共享噪声和 selection。
9. `DiT4DiT/training/train.py` 与 `DiT4DiT/training/trainer_utils/trainer_tools.py`：DeepSpeed、日志、checkpoint 和 resume。
10. `examples/EndoWAM/train_files/`：环境、preflight、launch 和 supervisor。

## Data contract

远端数据根目录的期望结构是：

```text
endowam_pseudo_z60/
├── ureter/
│   ├── data/chunk-000/
│   ├── videos/chunk-000/
│   └── meta/
├── ercp/
│   └── ...
└── esophagus/
    └── ...
```

每个子集至少必须包含 `meta/info.json`、`meta/stats_gr00t.json`、`data/chunk-000` 和 `videos/chunk-000`。当前已核对的样本数为：

| 子集 | 样本数 |
| --- | ---: |
| `ureter` | 406,612 |
| `ercp` | 407,617 |
| `esophagus` | 165,335 |
| 原始样本合计 | 979,564 |

三个 mixture weight 都是 `1.0`，而 loader 默认 `balance_dataset_weights=false`，所以三者以相同概率采样。按当前 `LeRobotMixtureDataset.__len__()` 规则，单个 epoch 的逻辑长度为 `3 × max(406612, 407617, 165335) = 1,222,851`；这会对较小子集进行重采样，并不表示磁盘上新增了样本。

字段契约：

| 模态 | LeRobot key | 模型形状/语义 |
| --- | --- | --- |
| 图像 | `video.endoscope` | 5 sparse frames，3 × 224 × 224 |
| 状态 | `state.endoscope_state` | 3 axes，值为 `-1/0/+1` |
| 动作 | `action.endoscope_cmd` | 64 × 3，值为 `-1/0/+1` |
| 语言 | `annotation.human.action.task_description` | task prompt |

必须保持以下不变量：

- `action_class_values: [-1, 0, 1]` 的顺序是 checkpoint 语义的一部分，不可随意重排。
- action/state 是原生类别语义值，不做均值方差归一化、阈值化或第二次离散化。
- episode 尾部不足 64 steps 的位置由 `action_mask` 屏蔽；送入世界模型前，padding 解码为 neutral command `0`。
- `action_horizon=64`、`future_action_window_size=63`、`action_dim=3`、`state_dim=3`、`valid_action_dim=3` 必须互相一致。
- `framework.name` 必须保持 `DiT4DiT`。`EndoWAM` 是数据/任务名，不是已注册 framework name。
- 离散 action head 会强制每个样本只计算一次 categorical loss；不要照搬连续 ActionDiT 的多次 diffusion target repetition。

如果数据 schema 有变化，需要同步修改 data config、mixture、YAML 的 `modality_keys`，并更新 `tests/test_discrete_dataset.py`。不要只在 YAML 中重命名 key。

## End-to-end code flow

```text
LeRobot subset
  -> DATASET_NAMED_MIXTURES["endowam_pseudo_z60"]
  -> EndoWAMEndoscopeDiscreteDataConfig
  -> dataset sample + action_mask
  -> DiT4DiT.forward()
       -> Cosmos25 policy/video features
       -> DiscreteActionHead categorical action loss
       -> action-conditioned Cosmos future-video loss
  -> total loss
  -> Accelerate + DeepSpeed ZeRO-2
  -> atomic full-state checkpoint + completion manifest
```

### Policy and discrete action head

Cosmos 提供视觉/语言 hidden representation。离散 head 使用 64 个可学习时间 query，通过 TransformerDecoder 读取 Cosmos hidden 和 state memory，输出 `(B, 64, 3, 3)` logits：batch、时间、动作轴、类别。

训练使用 mask-aware、可加权的 cross entropy。推理时：

- `K=1` 对每个时间和轴执行 argmax；
- `K>1` 从 categorical distribution 独立采样候选；
- 返回值被解码成 `-1/0/+1`，不是 class index。

当前配置的 per-axis class weights 来自数据分布。若重算权重，应记录脚本、数据版本和 class order；不可在未核对 class order 时复制数值。

### Action-conditioned world model

离散动作不会作为连续坐标直接投影。代码先把每个时间 × 轴编码为三分类 one-hot，再将整条轨迹投影为 Cosmos action-conditioning token。joint path 同一 optimization step 计算：

- `action_loss`：离散动作预测；
- `future_video_loss`：动作条件未来 latent flow matching；
- `total_loss = action_loss + future_video_loss_scaled`。

Stage 1 推理另外生成 reference future 和多条 candidate future，使用相同世界模型 seed/noise 做 latent cosine alignment，再选择最高分动作。

## Configuration that currently matters

ren5 主配置中的有效默认值：

```yaml
framework:
  cosmos25:
    max_sequence_length: 64
    extract_layer: 17
    training: joint
    future_num_inference_steps: 1
  action_model:
    action_head_type: discrete
    action_horizon: 64
    action_dim: 3
    state_dim: 3
    num_action_classes: 3
    action_class_values: [-1, 0, 1]

datasets:
  vla_data:
    per_device_batch_size: 3
    video_delta_indices: [0, 16, 32, 48, 63]
    num_workers: 2

trainer:
  max_train_steps: 100000
  num_warmup_steps: 500
  save_interval: 5000
  eval_interval: 100
  gradient_accumulation_steps: 1
  checkpoint_exclude_frozen_parameters: true
```

五种已见 task prompt 经 Cosmos chat template 后最长 37 tokens，因此 `max_sequence_length=64` 留有余量，并显著减少 cross-attention padding。它不会改变参数 shape 或 checkpoint shape。如果 prompt 集合发生变化，应重新统计 token 长度后再决定是否保持 64。

`attn_implementation: sdpa` 当前只记录在 YAML，`Cosmos25` loader 尚未把这个字段传给底层 transformer。仅修改该字段或安装 `flash-attn` 不会自动切换 attention backend；如需启用，必须补齐 loader plumbing、环境依赖和数值/速度测试。

## ren5 performance rationale

ren5 当前只有 CUDA indices 0 和 1 两张 RTX 3090 可用于训练；GPU 2 有 NVML/PCIe 故障。两个健康 GPU 间没有可用 P2P，因此旧 ZeRO-3 方案通信代价很高。已验证的折中是：

- BF16；
- ZeRO-2，optimizer state CPU offload；
- per-device batch 3；gradient accumulation 1；global batch 6；
- `OMP_NUM_THREADS=6`，其余 BLAS thread pool 为 1；
- Cosmos gradient checkpointing 开启；
- 数据 workers 每 rank 2。

该配方测得约 4.9–5.2 秒/step、峰值显存约 21.82 GiB；旧 global-batch-6 ZeRO-3 约 10.1 秒/step。数据加载约 0.001–0.003 秒，不是当前瓶颈。

模型计算较重的根本原因是 joint path 含两次 Cosmos forward，并且 gradient checkpointing 在 backward 重算。优化时应先 profile 这两部分，不要默认增加 DataLoader workers。

任何速度数字都依赖 commit、驱动、温度和同时运行的进程。改配置后重新做小规模 benchmark，不能把上面的记录当成硬保证。

## Checkpoint and resume semantics

ren5 配置冻结 Cosmos text encoder 和 VAE，并设置：

```yaml
checkpoint_exclude_frozen_parameters: true
save_consolidated_checkpoints: false
```

因此周期 checkpoint 保留训练所需的 trainable model state、optimizer、scheduler、RNG 和 dataloader position，但不重复保存冻结权重。恢复时冻结权重从配置锁定的本地 Cosmos base model 重新加载，DeepSpeed module 使用 non-strict load 接受这些缺失 frozen keys。

这带来三个重要约束：

1. base model 目录和 revision 必须保持可用且内容一致；checkpoint 不能脱离它独立恢复。
2. manifest 中的 `exclude_frozen_parameters` 必须与当前 run config 一致，否则自动恢复会拒绝该 checkpoint。
3. `save_consolidated_checkpoints=false` 时不会生成每个周期的单文件推理权重；full-state 目录才是恢复来源。

checkpoint 发布顺序：

```text
checkpoints/
├── steps_5000_state/            # 完成后由 .incomplete 原子改名
└── steps_5000_complete.json     # 最后发布；auto-resume 的唯一有效标志
```

`RESUME=auto` 只识别带合法 `steps_N_complete.json` 的最新 state。首个周期 checkpoint 是 step 5,000；在此之前停止会从 step 0 重启。实测单个排除 frozen weights 后的 checkpoint 约 28 GB，保存约 1 分 46 秒；20 个周期 checkpoint 约 560 GB，尚未包含 final model、日志、缓存和安全余量。当前代码没有自动轮换旧 checkpoint，正式 100k run 建议至少预留 750 GB，或先实现并测试 retention policy。

## Safe modification workflow

以下命令是交接流程示例；本文档更新本身没有启动训练。

1. 从精确 commit 建新分支，保持 checkpoint、数据、缓存和 secrets 在 Git 之外。
2. 修改前记录配置不变量和预期 shape。
3. 运行离散路径 CPU tests：

   ```bash
   pytest -q \
     tests/test_discrete_action_head.py \
     tests/test_discrete_dataset.py \
     tests/test_cosmos_discrete_action_condition.py \
     tests/test_stage1.py \
     tests/test_stage1_world_model.py
   ```

4. 在 ren5 上核对数据、base model、环境、GPU、磁盘、Git commit 和 resume policy。
5. 用新的 `RUN_ID` 做短 smoke/benchmark；不要覆盖正式 run。
6. 检查 finite losses、显存、step time、evaluation 和一次 checkpoint round-trip。
7. 只有 smoke + resume 都通过后，才在 supervisor + tmux 中开始正式训练。

### Where to change common features

| 需求 | 主要修改点 | 必须同步检查 |
| --- | --- | --- |
| 新增/重命名数据字段 | `data_config.py`、YAML `modality_keys` | mixture、dataset tests、remote metadata |
| 增删数据子集或改采样权重 | `mixtures.py` | 数据完整性 preflight、统计口径 |
| 修改 action class/value | discrete head config/implementation | class weights、one-hot world condition、checkpoint compatibility |
| 修改 horizon 或 action dim | YAML、data config、discrete head | masks、projector input shape、checkpoint incompatible change |
| 修改 Cosmos prompt/token length | `Cosmos25.py`、YAML | 全部 prompts 的 token length、显存/速度 |
| 修改 Stage 1 selector | `stage1.py` | shared-noise tests、candidate batch layout |
| 调整分布式/显存策略 | DeepSpeed JSON、Accelerate YAML、runner | global batch、optimizer state、checkpoint round-trip |
| 修改 checkpoint 格式 | `train.py`、`trainer_tools.py` | atomic manifest、旧 checkpoint migration、multi-rank tests |
| 增加任务级评估 | `examples/EndoWAM/eval_files/` | evaluator/environment version、成功率协议 |

改变 horizon、action dim、class order 或 action projector 输入形状通常与旧 checkpoint 不兼容。应使用新 `RUN_ID`，不要伪装成 full-state resume。

## Remote deployment reference

以下是 ren5 当前部署的默认位置，仅用于该主机；代码本身应继续支持通过环境变量覆盖：

```text
repository  /mnt/data-hdd2/ljs/DiT4DiT
conda env   /mnt/data-hdd2/ljs/.conda/envs/dit4dit-ren5
dataset     /mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60
base model  /mnt/data-hdd2/ljs/models/Cosmos-Predict2.5-2B
run root    /mnt/data-hdd3/ljs/experiments/DiT4DiT
```

Drive 凭据只应存在于远端用户的 `rclone` 配置；不得把账号、token 或配置文件提交到仓库。大数据应从 Drive 直接同步到远端 cache，不经过本地 Mac。

## Known operational traps

- 不要使用 GPU 2，除非硬件已修复并重新通过 CUDA、NVML 和 NCCL 检查。
- `run_endowam_ren5_2x3090.sh` 是直接 runner；长期任务应由 `supervise_endowam_ren5.sh` 启动。
- supervisor 用 `flock` 防止同一个 `RUN_ID` 重复启动，并把训练语义参数锁在 `supervisor_logs/launch_config.env`。不同参数不能静默续跑同一 run。
- supervisor 默认最多尝试 3 次；OOM、依赖缺失、脏工作树等确定性错误不会自动重试。
- 人工 `Ctrl-C` 时 `tee` 也可能收到信号并打印 logging failure；中断仍不会自动重试。停止后要用进程和 GPU 状态确认所有 rank 已退出。
- `WANDB_MODE=offline` 是默认值。切换 online 前在主机环境完成认证，不要把 key 写进脚本。
- `max_sequence_length=64` 只适用于当前 prompt 集；新增长 prompt 后必须重新验证。
- EndoWAM 当前只有 offline action accuracy，没有仓库内完整 task-level online evaluator。

## Handoff checklist

交给下一位开发者时至少附上：

- 精确 Git commit 和分支；
- 修改目的，以及哪些数据/shape/checkpoint 不变量被改变；
- 使用的 config、DeepSpeed config 和完整 launch command；
- 数据路径、数据版本/统计和 base model revision；
- 单元测试、smoke、峰值显存和 step-time 结果；
- run ID、输出目录、最新完整 manifest 和 resume 方式；
- 已知问题，以及是否存在仍在运行的 tmux/supervisor/worker。

不要只交接一个 `.pt` 文件或一段终端截图；可复现修改需要代码、配置、日志和 checkpoint policy 一起记录。
