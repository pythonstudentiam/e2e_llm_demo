# 10 — VS Code

**Notebook:** `notebooks/local/11_client_api.ipynb`
**Config:** `continue/config.yaml`
**Client:** `clients/chat_cli.py`

## What this stage does

Puts the model behind the four interfaces you'd actually use it through.

## Setup

```powershell
.\scripts\serve.ps1                                       # terminal 1
Copy-Item continue\config.yaml $HOME\.continue\config.yaml
```

Install the **Continue** extension, reload VS Code, open the sidebar, pick
**tinyllm (local)**.

---

## Set your expectations first

**A 15.7M-parameter TinyStories model cannot write code.** It has never seen any.
It cannot answer factual questions, do arithmetic, or follow multi-step
instructions.

So in `continue/config.yaml` it is given exactly one role:

```yaml
roles:
  - chat
```

Assigning `autocomplete`, `edit` or `apply` would insert confident nonsense into
your files. The setup would *feel* broken while behaving exactly as a model this
size should — which is a bad way to end a project that otherwise worked.

**What's worth seeing is the path, not the output quality:**

```
VS Code → Continue → HTTP → llama-server → your GGUF → your weights
```

Ask it for a story in the sidebar. That it answers at all is the thing.

### The context problem

Continue normally attaches file and repository context to every request. With a
**512-token window** there's no room for it — the story would be pushed out of the
context before generation began. Hence `context: []` and a small `maxTokens`.

This is a real constraint of a tiny model, not a misconfiguration. It's also a
concrete illustration of why context length is one of the headline numbers on
every model release.

---

## The four interfaces

### 1. Continue sidebar
Chat panel inside the editor. Demonstrates the full IDE integration path.

### 2. Jupyter notebooks
`notebooks/local/` — GGUF internals, the quantization tradeoff, and the API
client, all on the torch-free kernel.

### 3. Python client
```powershell
python clients\chat_cli.py
```
Streaming chat via the `openai` SDK. There is nothing model-specific in it — point
`--base-url` at OpenAI and the script keeps working. That interchangeability is
the whole value of the API standard.

### 4. Terminal
```powershell
.\vendor\llamacpp\llama-cli.exe -m .\models\tinyllm-Q8_0.gguf -cnv
```

---

## Why "OpenAI-compatible" matters more than it sounds

`llama-server` implements someone else's API, and that's the reason the final
stage of this project is three lines of YAML rather than an integration effort.

The lesson generalizes past this project: **the interface is the leverage.** Once
a tool speaks a widely-implemented API, every client that speaks it works
immediately, and swapping the model behind it costs a URL change. When you want an
assistant that can actually code, you don't rewrite anything — you serve a
different GGUF on a different port and add an entry to the same config file. The
commented block at the bottom of `continue/config.yaml` shows exactly that, using
Qwen2.5-Coder.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `finish_reason: length` every time | EOS id in the GGUF isn't `<|im_end|>`. Re-export with `chat_model=True` (stage 7) |
| Replies ignore the chat format | `--jinja` missing, or no chat template in the GGUF. Check with notebook 09 |
| Continue shows nothing | server not running, or `model:` doesn't match `--alias` in `serve.ps1` |
| Output degrades after a few turns | context full at 512 tokens; the server is truncating the front of the prompt |
| Port already in use | `serve.ps1 -Port 8081` |

---

## The through-line

Trace a reply in the sidebar backwards:

```
the text in the sidebar
  ← Continue's HTTP request
  ← llama-server's sampler
  ← your Q8_0 GGUF
  ← your f16 GGUF
  ← your Hugging Face repo
  ← your SFT run
  ← your 45-minute pretraining run
  ← 328M tokens through 8 transformer blocks
  ← random initialization
```

Every arrow is a stage in this repo.

---

## Where to go next

- **Make it better** — raise `max_steps` in `config.py` and retrain. Everything
  downstream follows automatically.
- **Make it bigger** — `hidden_size`, `num_hidden_layers`, `intermediate_size`.
  Watch how the parameter breakdown and time estimate move.
- **Different data** — swap `DataConfig.dataset_id`. The tokenizer refits, and you
  learn how much the domain mattered.
- **Preference tuning** — [DPO](https://arxiv.org/abs/2305.18290) is the natural
  next step conceptually, though this model is too small for it to mean much.
- **Fine-tune a real model** — apply stages 6–10 to Qwen2.5-0.5B or SmolLM2-135M.
  The packaging, conversion, quantization and serving code all work unchanged,
  which is the best evidence that what you built is a pipeline rather than a
  one-off.

**Back to:** [00 — Overview](00-overview.md)
