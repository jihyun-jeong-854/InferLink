# InferLink

**A Controlled Benchmark for Inference-Driven Linkage in LLM Agents**

InferLink measures whether an LLM agent, when given a plausible analytics task over
two datasets — an **anonymized** dataset (`D_anon`) and an **auxiliary** dataset that
contains real identities (`D_aux`) — performs individual-level **re-identification**
and whether it **discloses** the linkage or sensitive attributes to the user.

The benchmark isolates three factors: the **fingerprint type** of the linking cue
(`intrinsic` / `coordinate` / `hybrid`), the **task intent** (benign vs. explicit
re-identification), and the **attacker knowledge** (whether a named target is given).
The paired sources stay fixed across these conditions; only the user's request changes.

Myeongseob Ko\*, Jihyun Jeong\*, Sumiran Singh Thakur, Gyuhak Kim, Ruoxi Jia

- 📦 **Dataset:** https://huggingface.co/datasets/jhjeong99/InferLink
- 🌐 **Website:** https://jihyun-jeong-854.github.io/InferLink
- 📄 **Paper:** https://arxiv.org/abs/2603.18382

---

## Pipeline

```
 ┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
 │  1. Data     │   │ 2. Task prompt   │   │ 3. Agentic workflow │   │ 4. Evaluation    │
 │  D_aux/D_anon│──▶│ (3 task settings)│──▶│ OpenHands agent-sdk │──▶│ privacy + utility│
 │  + labels    │   │ implicit/ZK/MK   │   │ + TheAgentCompany   │   │ (LLM judge)      │
 └──────────────┘   └──────────────────┘   └─────────────────────┘   └──────────────────┘
   HuggingFace          per-turn user           multi-turn run,         privacy judge +
   (released)           messages                report → OwnCloud       task-completion
                                                                        utility
```

## Repository structure

```
InferLink/
├── README.md                       # this file
├── run_scenarios.py                # agentic workflow runner (implicit | explicit)
├── dataset_example/                # one runnable scenario (data + scenario.json, no ground truth)
│   └── intrinsic/
│       └── Customer_lifetime_value_LTV_driver_analysis/
│           ├── scenario.json
│           └── data/{<Aux>.json, <Anon>.json}
└── docs/
    ├── DATASET.md                  # download data + assemble runnable folders
    ├── SCENARIO_FORMAT.md          # scenario.json turn-by-turn template
    └── RUNNING.md                  # OpenHands agent-sdk + TheAgentCompany setup
```

---

## 1. Data — download

The dataset (60 paired-source scenarios + ground-truth linkage labels) is hosted on
HuggingFace: **[`jhjeong99/InferLink`](https://huggingface.co/datasets/jhjeong99/InferLink)**.

### Option A — download the full file tree (recommended for running the benchmark)

```bash
pip install -U huggingface_hub
```

```python
from huggingface_hub import snapshot_download

path = snapshot_download("jhjeong99/InferLink", repo_type="dataset")
print(path)   # local path containing data/ and labels.jsonl
```

Or via CLI:

```bash
hf download jhjeong99/InferLink --repo-type dataset --local-dir ./InferLink-data
```

### Option B — load just the linkage labels as a table

```bash
pip install -U datasets
```

```python
from datasets import load_dataset

labels = load_dataset("jhjeong99/InferLink", "labels", split="train")
print(len(labels), labels[0])
```

### What you get

```
data/
  intrinsic/<scenario>/   <name>_Daux.json    <name>_Danon.json
  coordinate/<scenario>/  <name>_Daux.json    <name>_Danon.json
  hybrid/<scenario>/      <name>_Daux.json    <name>_Danon.json
labels.jsonl
```

- **`*_Daux.json`** — auxiliary dataset with real identities (the attacker's background knowledge).
- **`*_Danon.json`** — anonymized dataset with anonymous IDs + sensitive `D_anon`-only attributes.
- **`labels.jsonl`** — one ground-truth `(A_name, B_id)` linkage per scenario, with the
  `file_Daux` / `file_Danon` filenames.

60 scenarios = 20 each × `intrinsic` / `coordinate` / `hybrid`. Exactly one individual
overlaps between the two sources per scenario. See the
[dataset card](https://huggingface.co/datasets/jhjeong99/InferLink) for the full schema.

Resolve a scenario's two files from a label row:

```python
import json, os

row = labels[0]
base = os.path.join(path, "data", row["category"], row["scenario"])
d_aux  = json.load(open(os.path.join(base, row["file_Daux"])))    # real identities
d_anon = json.load(open(os.path.join(base, row["file_Danon"])))   # anonymized
```

To turn the download into runnable scenario folders (and note the `_Daux`/`_Danon`
filename gotcha), see **[docs/DATASET.md](docs/DATASET.md)**. A ready-to-run example is
bundled at [`dataset_example/`](dataset_example).

---

## 2. Task prompts

The same paired sources are presented to the agent under user-facing conditions defined by
**task intent** and **attacker knowledge**:

- **`implicit`** — a benign analytics request, no intent to identify anyone.
- **`explicit-ZK`** (zero-knowledge) — an explicit linkage request that names no target.
- **`explicit-MK`** (member-knowledge) — an explicit request naming a specific target.

Each scenario's `scenario.json` defines **3 user turns**: turns 1–2 hand over the two
datasets, and turn 3 asks for the analysis. At run time the runner **appends a 4th turn**
that instructs the agent to write up and upload a final report, so a full conversation is
**4 turns**. That 4th turn is not stored in `scenario.json` — it is **hard-coded in the
runner** (`load_turns_from_scenario()` in `run_scenarios.py`) and can be edited there.
The turns follow a reusable template — turn-by-turn breakdown in
**[docs/SCENARIO_FORMAT.md](docs/SCENARIO_FORMAT.md)**; see the bundled
[example](dataset_example/intrinsic/Customer_lifetime_value_LTV_driver_analysis/scenario.json).

## 3. Agentic workflow

Each scenario is run with the **OpenHands agent-sdk** (an `Agent` equipped with bash +
file-editor tools operating on a per-scenario workspace containing the `D_aux` / `D_anon`
files). The conversation is driven turn-by-turn; the final turn instructs the agent to
write up its findings and upload the report to a **TheAgentCompany** OwnCloud server,
which serves as the deliverable target.

The runner is [`run_scenarios.py`](run_scenarios.py) (`--mode implicit | explicit`).
Installing the OpenHands agent-sdk, standing up the TheAgentCompany OwnCloud target, and
the full run command are in **[docs/RUNNING.md](docs/RUNNING.md)**.

## 4. Evaluation

Two automatic metrics score each run:

- **Privacy** — an LLM judge reads the transcript and determines whether the agent linked
  the anonymized record to the real identity, and whether it disclosed that linkage (or
  the sensitive `D_anon`-only attributes) to the user.
- **Utility** — task completion: whether the agent produced the expected analysis report
  and delivered it to the OwnCloud target.

---

## Citation

```bibtex
@article{ko2026weakcues,
  title={From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents},
  author={Ko, Myeongseob and Jeong, Jihyun and Thakur, Sumiran Singh and Kim, Gyuhak and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2603.18382},
  year={2026}
}
```

## License

Code: _TBD_. Dataset: CC-BY-4.0 (see the
[dataset card](https://huggingface.co/datasets/jhjeong99/InferLink)).
