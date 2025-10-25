# Data Directory Overview

This README describes how data is organized for bias control experiments, the conventions used across scenarios, and how to add new data. It is self-contained and does not depend on any other data README files.

## Structure

- Subfolders by bias type (each contains JSON scenarios):
  - `authority/`: `authority_MilgramS.json`, `authority_StanPri.json`
  - `bandwagon/`: `bandwagon_Asch.json`, `bandwagon_Hotel.json`
  - `confirmation/`: `confirmation_Wason.json`, `confirmation_BiasInfo.json`
  - `framing/`: `framing_Asian.json`, `framing_Invest.json` (may include `experiment_results.csv` samples)

There is no required CSV or template file in the root. Scenarios can be authored directly as JSON files in the appropriate subfolder.

## Core conventions

- MCQ options must use labels `A, B, C, D, E`.
- Likert weights used by the code (for a single scalar score):
  - A = 4, B = 3, C = 2, D = 1, E = 0
- Each scenario defines either Q&A or MCQ prompts. For MCQ, provide the five option texts with clear semantic ordering from strong to weak (or positive to negative) so the A–E mapping is consistent.
- Keep placeholder names clear and minimal if you use them (e.g., `[Authority Source]`, `[Statement]`, `[Scenario]`, `[Rule]`). Replace them with concrete values when generating instances.

## Metrics (used by experiments)

- Probability normalization across options:
  - P(A) + P(B) + P(C) + P(D) + P(E) = 1
- Weighted Likert score (scalar):
  - Weighted = 4·P(A) + 3·P(B) + 2·P(C) + 1·P(D) + 0·P(E)
- Advanced evaluation (monotonicity, smoothness, efficacy) is implemented in code under `control/evaluate_metrics.py` and plotting helpers in `control/base.py`.

## How experiments consume data

- Prompt-based and API experiments build prompts from scenario text and compute the next-token probabilities for A–E (or frequency of extracted choices for API models). Probabilities are summed over token variants mapped to A–E.
- RepE experiments use the same MCQ prompts but modify internal activations to steer outputs over a range of control coefficients.
- All experiments expect the A–E labels to be present and consistently ordered.

## Minimal workflow

1) Author or select a scenario JSON in the appropriate subfolder (e.g., `authority/authority_MilgramS.json`).
2) Ensure it defines either a Q&A or MCQ prompt; for MCQ include A–E option texts.
3) Run a control experiment (Prompt / RepE / API) to obtain A–E probabilities or choice frequencies.
4) Compute the weighted Likert score (and optional advanced metrics) and inspect saved plots/JSON outputs.

## Adding new data

- Create a new JSON file under the matching bias subfolder. Include:
  - A descriptive name (e.g., `bandwagon_NewScenario.json`).
  - Clear prompt text and, for MCQ, the five option texts with A–E mapping.
  - Any placeholders you plan to fill during generation.
- Keep the A–E semantics consistent across scenarios so metrics are comparable.

## Notes

- If you previously relied on a centralized CSV of statements or a template file, simply embed the necessary content directly in each scenario JSON. No separate CSV/JSON templates are required.
- The experiments save plots and result summaries to the configured output folders in the control code.
