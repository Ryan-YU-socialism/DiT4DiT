# EndoWAM `endowam_pseudo_z60` training

When H800 is unavailable, the recommended balanced AutoDL setup is one host
with **4 x RTX PRO 6000 96GB**.  The recipe uses the CUDA 12.8 build of PyTorch
2.7, Triton 3.3, BF16, DeepSpeed ZeRO-2, micro-batch 4 per GPU, and gradient
accumulation 2, for a global batch size of 32.  The GPU exposes a PCIe Gen 5
interface; inspect the actual host with `nvidia-smi topo -m` and prefer a
machine that reports direct GPU peer-to-peer paths.

If only two RTX PRO 6000 cards are available on the same host, use the provided
2-GPU launcher.  It raises gradient accumulation to 4 and keeps the same global
batch size and learning-rate schedule.  It is slower but remains checkpoint-
compatible with the 4-GPU run.

Use at least 300GB of persistent data-disk space for the repository, the
Cosmos-Predict2.5-2B weights, caches, logs, and multiple checkpoints.  The
Drive dataset itself contains 515 videos and about 979,564 frames; the video
payload is only about 2.26GiB, but checkpoints dominate disk usage.

## Remote preparation

Keep Google credentials in `rclone`'s user configuration, never in this
repository.  From the AutoDL host, sync Drive directly into the training cache:

```bash
bash examples/EndoWAM/train_files/sync_endowam_from_drive.sh \
  gdrive:endowam_pseudo_z60
```

Download the base model directly on the same host:

```bash
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir /root/autodl-tmp/models/Cosmos-Predict2.5-2B
```

Install the CUDA 12.8 build of PyTorch 2.7 before the repository requirements:

```bash
conda create -n dit4dit-stage1 python=3.10 -y
conda activate dit4dit-stage1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

## Launch

Run inside `tmux` so disconnecting from AutoDL does not stop training:

```bash
tmux new -s endowam
bash examples/EndoWAM/train_files/run_endowam_4xpro6000.sh
```

The script checks all three dataset subsets, the model directory, CUDA GPU
count, BF16 support, exact Git commit, output directory, and resume behavior
before launching.  It refuses a dirty Git worktree by default, uses offline
W&B logging, and never embeds an API key.  Set `WANDB_MODE=online` only after
authenticating W&B in the host environment.

Before committing to the full rental, run a 200-step throughput/memory check:

```bash
RUN_ID=dit4dit_endowam_smoke \
MAX_TRAIN_STEPS=200 SAVE_INTERVAL=200 EVAL_INTERVAL=200 \
bash examples/EndoWAM/train_files/run_endowam_4xpro6000.sh
```

Use the unchanged defaults for the full 80,000-step run after the smoke test
shows finite losses and no CUDA out-of-memory error.

For the two-card fallback, launch
`run_endowam_2xpro6000.sh`; gradient accumulation 4 preserves global batch 32.
If the 96GB card still cannot sustain micro-batch 4 in the installed Cosmos
build, use `PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4` on four cards
or `PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8` on two cards, keeping
global batch 32. `RESUME=auto` loads the newest model checkpoint from the same
run directory. The current trainer restores model weights and the scheduler
position, but initializes a fresh optimizer.
