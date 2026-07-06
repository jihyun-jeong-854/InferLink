# Running the agentic workflow

[`run_scenarios.py`](../run_scenarios.py) drives each scenario turn-by-turn with an
**OpenHands agent-sdk** agent (bash + file-editor tools over a per-scenario workspace),
then the final turn asks the agent to upload a report to a **TheAgentCompany** OwnCloud
server that stands in as the deliverable target.

You need two external pieces installed: **(A) OpenHands agent-sdk** and **(B) an OwnCloud
endpoint from TheAgentCompany**.

---

## A. OpenHands agent-sdk

The runner imports `openhands.sdk` / `openhands.tools` and expects the SDK checkout to sit
next to this repo (`sys.path.append('../agent-sdk')`).

**Prerequisites:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/) ≥ 0.8.13.

```bash
# next to this repository (so ../agent-sdk resolves)
git clone https://github.com/All-Hands-AI/agent-sdk.git
cd agent-sdk
make build                                   # installs openhands-sdk + openhands-tools
uv run python examples/01_hello_world.py     # smoke test
cd ..
```

Layout the runner assumes:

```
parent/
├── agent-sdk/        # cloned above
└── InferLink/        # this repo  (run_scenarios.py does sys.path.append('../agent-sdk'))
```

If you keep the SDK elsewhere, edit the `sys.path.append('../agent-sdk')` line in
`run_scenarios.py`.

## B. TheAgentCompany — OwnCloud deliverable target

The final turn tells the agent to upload its report to
`http://the-agent-company.com:8092` (OwnCloud), logging in with
`theagentcompany / theagentcompany`. Those services come from **TheAgentCompany**:

```bash
git clone https://github.com/TheAgentCompany/TheAgentCompany.git
```

Follow their setup to bring up the self-hosted services with Docker (OwnCloud is the only
one InferLink needs). Then make the URL in the runner reach your OwnCloud instance:

- **Hostname** — the runner uses the host `the-agent-company.com`. Point it at your server,
  e.g. add to `/etc/hosts`:  `127.0.0.1  the-agent-company.com`  (for a local Docker host).
- **Port / credentials** — defaults are `:8092` and `theagentcompany/theagentcompany`.
  If yours differ, edit them in `load_turns_from_scenario()` (the `final_turn` string) in
  `run_scenarios.py`.

> The OwnCloud target is only the *deliverable sink*. The privacy signal is judged from
> the transcript, so a misconfigured upload does not corrupt the privacy evaluation — but
> it does affect the utility (task-completion) score.

## C. Model provider credentials

Pick a provider and export its key (see `build_llm()` for the exact handling):

```bash
export OPENAI_API_KEY=...        # --provider openai
export ANTHROPIC_API_KEY=...     # --provider anthropic
export LITELLM_API_KEY=...       # --provider litellm  (optional: LITELLM_BASE_URL)
```

## D. Run

```bash
# from this repo root, with ../agent-sdk installed and OwnCloud reachable
uv run python run_scenarios.py \
    --mode implicit \
    --scenarios_root ./dataset_example/intrinsic \
    --outputs_path ./outputs \
    --provider openai \
    --model_name gpt-5
```

Key flags:

| Flag | Meaning |
|---|---|
| `--mode {implicit,explicit}` | task condition; only changes the final report-turn wording |
| `--scenarios_root` | a **category** dir whose subfolders each hold `scenario.json` + `data/` |
| `--provider` / `--model_name` | `openai` \| `anthropic` \| `litellm` and the model id |
| `--no_sparse` | zero sparse features (appends a `_nosp` suffix to output names) |
| `--skip-list FILE` | optional text file of scenario names to skip (one per line) |

## E. Outputs

```
outputs/
├── workspaces/<Scenario>/            # per-scenario agent workspace (data files copied in)
└── transcripts/<Scenario>_<model>.jsonl   # full LLM message log, one JSON per line
```

Runs are **resumable**: a scenario whose transcript already exists and is non-empty is
skipped. The transcripts are the input to the privacy/utility evaluation.
