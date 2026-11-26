# MiniMind — Local GPU User Guide

Train a ~64M-parameter language model (`minimind-3`) **from scratch** on a single
local GPU, then watch the loss / aux_loss / learning-rate curves live in
TensorBoard from your laptop.

This guide is tested against this repo and covers:

1. [Prerequisites & environment setup](#-prerequisites--environment-setup)
2. [Step-by-step commands](#-step-by-step-commands) *(the core of the guide)*
3. [What the Transformer trains on](#-what-the-transformer-trains-on)
4. [TensorBoard dashboard from localhost](#-tensorboard-dashboard-from-localhost)
5. [Optional extra training stages](#-optional-extra-training-stages)
6. [File layout & where outputs land](#-file-layout--where-outputs-land)
7. [Troubleshooting](#-troubleshooting)

> **Reference hardware:** a single NVIDIA **A30 (24 GB)** (or RTX 3090, A100, …).
> The "Zero" model reproduces in roughly **2–2.5 h** on one 3090-class GPU.
> All training commands are run **from the `trainer/` directory** unless stated otherwise.

---

## ⚙️ Prerequisites & Environment Setup

### Step 0.1 — Verify your GPU is visible

```bash
nvidia-smi
```

- ✅ **If you see your GPU** → jump to Step 0.2.
- ❌ **If it says `No devices were found` / `Command not found`** → the driver is
  missing or not loaded. On Ubuntu:

  ```bash
  sudo apt-get update
  sudo ubuntu-drivers autoinstall      # picks the right driver for your GPU
  sudo reboot
  nvidia-smi                            # re-check after reboot
  ```

> Note: the GPU may be physically present (visible via `lspci | grep -i nvidia`)
> yet invisible to `nvidia-smi` until the kernel driver is installed/loaded.

### Step 0.2 — Install PyTorch with CUDA support

Pick the `cuXXX` index that matches your `nvidia-smi` CUDA version
(e.g. `cu121` for CUDA 12.1). See <https://download.pytorch.org/whl/torch_stable.html>.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 0.3 — Install project dependencies

```bash
cd /users/mosafer/minimind
pip install -r requirements.txt
```

### Step 0.4 — Install TensorBoard (for the dashboard)

```bash
pip install tensorboard
```

### Step 0.5 — Confirm Python sees the GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected:

```
CUDA: True
GPU: NVIDIA A30
```

---

## 🚀 Step-by-Step Commands

> Run these **in order**. Each stage's output becomes the next stage's input.

### Step 1 — Download the training data

For the **fast "Zero" reproduction** (~2.3 h, 2.8 GB total), you only need the two
`_mini` files. Place them in `./dataset/`:

```bash
pip install modelscope
cd /users/mosafer/minimind
modelscope download --dataset gongjy/minimind_dataset \
  --include "pretrain_t2t_mini.jsonl" "sft_t2t_mini.jsonl" \
  --local_dir ./dataset
```

<details>
<summary><b>Alternative: Hugging Face mirror</b></summary>

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download jingyaogong/minimind_dataset \
  --include "pretrain_t2t_mini.jsonl" "sft_t2t_mini.jsonl" \
  --local-dir ./dataset
```
</details>

<details>
<summary><b>Full-quality reproduction (optional, ~24 GB)</b></summary>

For the full `minimind-3` main-branch model, additionally download:
`pretrain_t2t.jsonl` (10 GB) and `sft_t2t.jsonl` (14 GB), and point the scripts at
them with `--data_path`.

```bash
modelscope download --dataset gongjy/minimind_dataset \
  --include "pretrain_t2t.jsonl" "sft_t2t.jsonl" \
  --local_dir ./dataset
```
</details>

Verify the files landed:

```bash
ls -lh ./dataset/
# expect: pretrain_t2t_mini.jsonl  (~1.2GB)  sft_t2t_mini.jsonl  (~1.6GB)
```

---

### Step 2 — Pretraining (required)

Teaches the model language patterns via next-token prediction.

```bash
cd /users/mosafer/minimind/trainer
python train_pretrain.py --use_tensorboard
```

- Output weights: `../out/pretrain_768.pth`
- TensorBoard events: `../runs/MiniMind-Pretrain/`

**Variants:**

```bash
# Multi-GPU (replace N with your GPU count):
torchrun --nproc_per_node N train_pretrain.py --use_tensorboard

# Resume after an interruption (auto-detects last checkpoint):
python train_pretrain.py --use_tensorboard --from_resume 1
```

<details>
<summary><b>Quick sanity test of the pretrained model</b></summary>

```bash
cd ..
python eval_llm.py --weight pretrain
```
</details>

---

### Step 3 — Supervised Fine-Tuning (required)

Turns the pretrained base into a chat/instruction-following model. **Loads the
pretrain weights automatically** (`--from_weight pretrain` is the default).

```bash
# still in ./trainer
python train_full_sft.py --use_tensorboard
```

- Output weights: `../out/full_sft_768.pth`
- TensorBoard events: `../runs/MiniMind-Full-SFT/`

**Variants:**

```bash
torchrun --nproc_per_node N train_full_sft.py --use_tensorboard
python train_full_sft.py --use_tensorboard --from_resume 1
```

---

### Step 4 — Test the final model

```bash
cd /users/mosafer/minimind
python eval_llm.py --weight full_sft
```

`--weight` takes a prefix: `pretrain`, `full_sft`, etc.

---

## 📊 What the Transformer Trains On

Three stages, three data types (all JSONL, all in `./dataset/`):

| Stage | File(s) | Format | What it teaches |
|-------|---------|--------|-----------------|
| **Pretrain** | `pretrain_t2t(_mini).jsonl` | `{"text": "..."}` | Language + world knowledge via next-token prediction. Chinese–English mix. |
| **SFT** | `sft_t2t(_mini).jsonl` | `{"conversations":[{"role","content"}]}` | Instruction following, multi-turn chat, tool-calling, thinking. |
| **RL** (optional) | `dpo.jsonl`, `rlaif.jsonl`, `agent_rl.jsonl` | preference pairs / SFT-style prompts | Preference alignment (DPO), RLAIF (PPO/GRPO/CISPO), agentic tool use. |

**Pretrain data** — plain text, one `{"text": ...}` per line. Sources are general
text corpora, curated dialogue, and distillation corpora under permissive licenses
(Craftsman, Magpie-Align, etc.), all **cleaned, deduplicated, and length-controlled**.

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

**SFT data** — multi-turn dialogues with `system / user / assistant / tool` roles.
A large fraction is **synthetic data distilled from stronger models** (Qwen3,
DeepSeek-R1, R1-Distill-SFT, COIG, Step-3.5-Flash-SFT…), plus **~100K tool-calling
samples** generated by qwen3-4b and **reasoning/thinking** data.

```jsonl
{"conversations": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]}
{"conversations": [
  {"role": "system", "content": "# Tools ...", "tools": "[...]"},
  {"role": "user", "content": "把'你好世界'翻译成english"},
  {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",...}]"},
  {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
  {"role": "assistant", "content": "Hello World"}
]}
```

**In one line:** big clean raw text (zh/en) → high-quality instruction + distilled +
tool-calling dialogues → (optional) preference/RL data. Pretrain gives the language,
SFT shapes the chatbot, RL polishes the quality.

---

## 📈 TensorBoard Dashboard from localhost

The `--use_tensorboard` flag writes the metrics
(`loss`, `logits_loss`, `aux_loss`, `learning_rate`, `epoch_time`) to event files
under `runs/<project>/`. View them from your laptop via an SSH tunnel.

### 1. Start the dashboard (on the server, 2nd terminal)

```bash
cd /users/mosafer/minimind
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

`--host 0.0.0.0` is what lets the SSH tunnel reach it. Both the Pretrain and
Full-SFT runs appear as separate runs under the same `logdir`.

### 2. Port-forward from your laptop

```bash
# On YOUR local machine (not the server):
ssh -N -L 6006:localhost:6006 user@serverIP
```

### 3. Open it

Browse to **http://localhost:6006** — you'll see the pretraining loss curve and the
SFT loss curve (and aux_loss / learning_rate) as separate runs.

> The x-axis is the log-interval index (same convention as wandb/SwanLab), not raw
> optimizer steps. Curves read correctly; only the axis scale differs.

---

## 🧩 Optional Extra Training Stages

All run from `./trainer` (evaluation / serving from the repo root unless noted).

| Task | Command | Extra data needed |
|------|---------|-------------------|
| LoRA | `python train_lora.py` | a `lora_*.jsonl` in `./dataset` |
| RLHF / DPO | `python train_dpo.py` | `dpo.jsonl` |
| RLAIF (PPO/GRPO) | `python train_grpo.py` | `rlaif.jsonl` |
| Agentic RL | `python train_agent.py` | `agent_rl.jsonl` |
| White-box distillation | `python train_distillation.py` | a teacher model |
| Eval with LoRA | `python eval_llm.py --weight full_sft --lora_weight lora_medical` | — |
| Tool-call test | `python eval_toolcall.py --weight full_sft` | — |
| OpenAI-compatible API | `python scripts/serve_openai_api.py` | — |
| WebUI | `cd scripts && streamlit run web_demo.py` | — |
| Merge LoRA → full weights | `cd scripts && python convert_model.py` | — |

You can add `--use_tensorboard` to any of the `train_*.py` scripts that log metrics.

---

## 🗂️ File Layout & Where Outputs Land

```
/users/mosafer/minimind/
├── dataset/                    # ← training data goes here
│   ├── pretrain_t2t_mini.jsonl
│   └── sft_t2t_mini.jsonl
├── trainer/                    # ← run all training from here
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   └── tb_utils.py             # ← TensorBoard logger (added)
├── out/                        # ← produced model weights
│   ├── pretrain_768.pth
│   └── full_sft_768.pth
├── runs/                       # ← TensorBoard event files (added)
│   ├── MiniMind-Pretrain/
│   └── MiniMind-Full-SFT/
├── checkpoints/                # ← resume checkpoints (<name>_768_resume.pth)
├── model/                      # architecture + tokenizer (no weights)
└── scripts/                    # API server, WebUI, conversion
```

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---------|-----|
| `nvidia-smi` → *No devices were found* | Driver not installed/loaded: `sudo ubuntu-drivers autoinstall` then reboot. GPU may still show in `lspci`. |
| `torch.cuda.is_available()` → `False` | PyTorch installed without CUDA. Reinstall with the correct `--index-url .../cu121`. |
| `ModuleNotFoundError: tensorboard` | `pip install tensorboard` |
| TB shows no runs yet | Training hasn't hit the first `log_interval` (default 100 steps). Wait; TB auto-refreshes. |
| `Cannot connect to TensorBoard` from laptop | Ensure `--host 0.0.0.0 --port 6006` **and** the `ssh -L 6006:localhost:6006` tunnel are both up. |
| Out of VRAM | Lower `--batch_size` (pretrain default 32 → 16) and/or `--max_seq_len`. |
| Crash mid-run | Restart with `--from_resume 1` to continue from the last checkpoint. |

### Useful `train_pretrain.py` / `train_full_sft.py` flags

```
--batch_size 32          --max_seq_len 340/768   --learning_rate 5e-4 / 1e-5
--epochs 2               --accumulation_steps 8  --grad_clip 1.0
--log_interval 100       --save_interval 1000    --device cuda:0
--dtype bfloat16         --data_path ../dataset/...
--use_tensorboard        --tb_dir ../runs        --use_wandb   (mutually exclusive, TB wins)
--from_weight pretrain   --from_resume 1
```
