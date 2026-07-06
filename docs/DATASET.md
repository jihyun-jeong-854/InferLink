# Dataset — download & assemble a runnable scenario

This page explains how to get the paired-source data and lay it out in the folder
structure the runner ([`run_scenarios.py`](../run_scenarios.py)) expects.

A ready-to-run example is already bundled at
[`dataset_example/`](../dataset_example) — use it as the reference layout.

---

## 1. What is hosted where

| Artifact | Location | Notes |
|---|---|---|
| Paired data (`D_aux`, `D_anon`) | **HuggingFace** [`jhjeong99/InferLink`](https://huggingface.co/datasets/jhjeong99/InferLink) | 60 scenarios × 2 files |
| Ground-truth linkage labels | HuggingFace, `labels.jsonl` | one `(A_name, B_id)` per scenario |
| Task prompt (`scenario.json`) | **this repo** | see [SCENARIO_FORMAT.md](SCENARIO_FORMAT.md) |

> The HuggingFace release contains **only the data and labels** — not `scenario.json`.
> A runnable scenario folder = **HF data + a `scenario.json`**.

## 2. The layout the runner expects

`run_scenarios.py` scans the *immediate subfolders* of `--scenarios_root` for a
`scenario.json`, and copies everything under that folder's `data/` into the agent
workspace:

```
<scenarios_root>/                 # e.g. dataset_example/intrinsic
└── <Scenario_Name>/
    ├── scenario.json             # the multi-turn task prompt
    └── data/
        ├── <Aux_Source>.json     # D_aux — real identities (auxiliary knowledge)
        └── <Anon_Source>.json    # D_anon — anonymized records + sensitive attrs
```

Ground truth (`ground_truth.json`) is **not** needed to run — it is only used by the
[evaluation](RUNNING.md) step, and is intentionally omitted from the public example.

## 3. Download the data from HuggingFace

```bash
pip install -U huggingface_hub
hf download jhjeong99/InferLink --repo-type dataset --local-dir ./InferLink-data
```

You get:

```
InferLink-data/
├── labels.jsonl
└── data/{intrinsic,coordinate,hybrid}/<Scenario_Name>/
        ├── <stem>_Daux.json      # ← note the _Daux / _Danon suffix
        └── <stem>_Danon.json
```

## 4. Filename gotcha — strip the `_Daux` / `_Danon` suffix

On HuggingFace the two files carry a **`_Daux` / `_Danon` suffix** so the roles are
self-documenting. But `scenario.json` refers to each source by its **original filename
without the suffix** (e.g. turn 1 mentions `Internal_CRM_&_Order_Database.json`, while
HF stores `Internal_CRM_&_Order_Database_Daux.json`). When assembling a runnable folder,
rename the files back:

```
Internal_CRM_&_Order_Database_Daux.json   →  Internal_CRM_&_Order_Database.json
Anonymous_..._Review_Platform_Logs_Danon.json  →  Anonymous_..._Review_Platform_Logs.json
```

Reconstruction script (HF download + your `scenario.json` files → runnable tree):

```python
import json, os, re, shutil

HF   = "./InferLink-data"                 # from step 3
DST  = "./scenarios_run"                  # runnable tree to build
STRIP = re.compile(r"_(Daux|Danon)(?=\.json$)")

for line in open(os.path.join(HF, "labels.jsonl")):
    row = json.loads(line)
    src = os.path.join(HF, "data", row["category"], row["scenario"])
    dst = os.path.join(DST, row["category"], row["scenario"], "data")
    os.makedirs(dst, exist_ok=True)
    for f in (row["file_Daux"], row["file_Danon"]):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, STRIP.sub("", f)))
    # then drop the scenario's scenario.json next to data/  (see SCENARIO_FORMAT.md)
```

After this, add each scenario's `scenario.json` alongside its `data/` folder and point
`--scenarios_root` at a category directory (e.g. `./scenarios_run/intrinsic`).

## 5. Which file is which (from a label row)

```python
row["file_Daux"]   # → real identities   (D_aux)
row["file_Danon"]  # → anonymized records (D_anon)
row["A_name"]      # ground-truth real identity in D_aux
row["B_id"]        # ground-truth anonymous id in D_anon that refers to the same person
```

Exactly **one** individual overlaps between `D_aux` and `D_anon` per scenario; the
benchmark measures whether the agent finds and discloses that link. See the
[dataset card](https://huggingface.co/datasets/jhjeong99/InferLink) for the full schema.
