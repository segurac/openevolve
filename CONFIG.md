# OpenEvolve Configuration Guide

This document collects every configuration flag that OpenEvolve currently understands and explains how they influence the system. Use it in combination with `configs/default_config.yaml`, which contains the canonical defaults and inline comments.

- **Where configs live**: place your YAML next to the initial program or reuse `configs/default_config.yaml`. Every example in `examples/*/config*.yaml` only overrides a subset of the values listed below.
- **How configs are loaded**: `openevolve-run.py` (or `python -m openevolve.cli`) accepts `--config path/to/config.yaml`. Command-line flags such as `--iterations` or `--primary-model` override the YAML after it is loaded.
- **Environment variables**: when a field is left `null`, OpenEvolve falls back to environment values (e.g. `OPENAI_API_KEY`, `OPENAI_API_BASE`) or auto-detects from the initial program.

---

## 1. Top-Level Run Settings (`Config`)

| Key | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | `10000` | Upper bound on evolution steps unless `--iterations` overrides it.
| `checkpoint_interval` | `100` | Save the database + best program every N iterations.
| `log_level` | `"INFO"` | Global log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
| `log_dir` | `null` | Custom directory for log files (defaults to `<output_dir>/logs`).
| `random_seed` | `42` | Master seed; propagated to the database and LLM ensembles for reproducibility.
| `language` | `null` | Programming language hint for prompts. When `null`, inferred from the initial program contents.
| `file_suffix` | `.py` | Extension used when writing temporary evolved programs. Overridden automatically from the initial file.
| `diff_based_evolution` | `true` | `true` keeps diffs anchored to the parent program; `false` expects full rewrites. Several example configs still include `allow_full_rewrites`; this legacy field is ignored, so use `diff_based_evolution: false` instead.
| `max_code_length` | `10000` | Hard limit on generated source length; responses longer than this are discarded.
| `early_stopping_patience` | `null` | When set, stop after this many iterations without a qualifying improvement (see below).
| `convergence_threshold` | `0.001` | Minimum delta on the tracked metric before the early-stopping patience resets.
| `early_stopping_metric` | `"combined_score"` | Metric pulled from evaluator results for the convergence check (falls back to the average of numeric metrics if absent).
| `max_tasks_per_child` | `null` | For Python ≥3.11 you can cap how many programs each worker process evaluates before it respawns. Helps control GPU/CPU resource leaks.

### Early stopping behaviour
When `early_stopping_patience` is non-null, the process stops if the target metric fails to improve by at least `convergence_threshold` for that many iterations. These checks are performed in `ProcessParallelController.run_evolution` while results stream back from workers.

---

## 2. LLM Settings (`llm` block)

The LLM section configures both the ensemble that proposes mutations and (optionally) a separate ensemble that provides evaluation feedback.

### 2.1 Ensemble-level options (`LLMConfig`)

| Key | Default | Notes |
| --- | --- | --- |
| `api_base` | `"https://api.openai.com/v1"` | Base URL shared by every model unless overridden per model. Examples often point this to local vLLM or Gemini emulation endpoints.
| `api_key` | `null` | Falls back to `OPENAI_API_KEY` if omitted.
| `models` | `[]` | List of model entries (see below). If empty, `primary_model`/`secondary_model` fields can be used for backward compatibility.
| `evaluator_models` | `[]` | Optional ensemble used only for evaluator LLM feedback. Defaults to the same list as `models` when left empty.
| `temperature`, `top_p`, `max_tokens` | `0.7`, `0.95`, `4096` | Shared generation parameters unless a model overrides them.
| `timeout`, `retries`, `retry_delay` | `60`, `3`, `5` | HTTP timeout/retry policy applied to every LLM request.
| `system_message` | `"system_message"` | Default system prompt passed to models when the prompt sampler does not override it.
| `reasoning_effort` | `null` | Enables vendor-specific reasoning knobs (e.g. OpenAI's `reasoning_effort` for o1-style models). Any string value is forwarded with every call.
| `primary_model`, `secondary_model` | `null` | Legacy shortcut to define up to two models without a `models` array. Use `primary_model_weight` / `secondary_model_weight` to set sampling weights.

### 2.2 Model-level entries (`LLMModelConfig`)
Each item inside `llm.models` has the following knobs:

| Field | Description |
| --- | --- |
| `name` | Required model identifier understood by your API endpoint.
| `weight` | Relative probability of picking this model inside the ensemble. Weights are normalized automatically.
| `api_base`, `api_key` | Optional per-model overrides.
| `init_client` | Advanced hook that receives the model config and returns an object implementing `LLMInterface`. Lets you plug in custom client classes beyond `openevolve.llm.openai.OpenAILLM`.
| `system_message`, `temperature`, `top_p`, `max_tokens`, `timeout`, `retries`, `retry_delay` | Override their ensemble counterparts for this model only.
| `random_seed` | Forces deterministic sampling order for this model.
| `reasoning_effort` | Overrides the ensemble’s `reasoning_effort`.

### 2.3 Evaluator feedback LLMs
If `evaluator.use_llm_feedback` is `true`, OpenEvolve requests additional quality assessments from `llm.evaluator_models`. Their scores are merged into evaluator metrics using `llm_feedback_weight`.

---

## 3. Prompt Builder (`prompt` block)

| Key | Default | Purpose |
| --- | --- | --- |
| `template_dir` | `null` | Directory containing custom prompt templates. When omitted, the built-in templates under `openevolve/prompt/templates` are used.
| `system_message` | Intelligent default | The primary system prompt for the evolution LLM. Can be literal text or the name of a template file in `template_dir`.
| `evaluator_system_message` | `"evaluator_system_message"` | Template used for evaluator LLM interactions.
| `num_top_programs` | `3` | Number of best programs injected into the prompt for inspiration.
| `num_diverse_programs` | `2` | Number of diverse programs (by MAP-Elites bins) shown to the LLM to encourage exploration.
| `use_template_stochasticity` | `true` | Enables random phrase swaps in the prompt to avoid repetition.
| `template_variations` | `{}` | Mapping of placeholder → list of alternative phrasings. Placeholders are referenced inside template files as `{placeholder}`.
| `use_meta_prompting`, `meta_prompt_weight` | `false`, `0.1` | Reserved for future meta-prompt features (currently no effect).
| `include_artifacts` | `true` | Inserts evaluator artifacts (stderr, profiling data, etc.) into the next prompt when available.
| `max_artifact_bytes` | `20 KB` | Size limit before artifacts are truncated.
| `artifact_security_filter` | `true` | Censors obvious secrets or tokens from artifacts before they enter prompts.
| `suggest_simplification_after_chars` | `500` | Prompts the LLM to consider simplifying bloated programs.
| `include_changes_under_chars` | `100` | Include change descriptions if short enough; avoids dumping large diffs into features.
| `concise_implementation_max_lines` / `comprehensive_implementation_min_lines` | `10` / `50` | Heuristics used when labelling programs as “concise” or “comprehensive” in the prompt context.
| `code_length_threshold` | `null` | Deprecated alias for `suggest_simplification_after_chars`.

Artifacts are only rendered if both `prompt.include_artifacts` and `evaluator.enable_artifacts` (and the `ENABLE_ARTIFACTS` env var) allow it.

---

## 4. Program Database & Diversity (`database` block)

### 4.1 Storage & logging

| Key | Default | Description |
| --- | --- | --- |
| `db_path` | `null` | Directory to persist the database. When left `null`, the run stays in-memory until a checkpoint is written.
| `in_memory` | `true` | Set `false` to always persist to `db_path` (slower but durable).
| `log_prompts` | `true` | When `true`, prompts + responses are stored per program under `programs/<id>.json`.
| `artifacts_base_path` | `null` | Custom root directory for evaluation artifacts (defaults to `<db_path>/artifacts`).
| `artifact_size_threshold` | `32768` | Artifacts larger than this are written to disk instead of being embedded in the DB JSON.
| `cleanup_old_artifacts` / `artifact_retention_days` | `true` / `30` | Whether to prune artifact folders older than the retention window when checkpoints run.
| `random_seed` | `42` | Controls deterministic parent sampling when drawing inspiration programs.

### 4.2 Population & selection knobs

| Key | Default | Description |
| --- | --- | --- |
| `population_size` | `1000` | Maximum number of non-archived programs kept per island.
| `archive_size` | `100` | Number of elite slots maintained independently of the population limit.
| `num_islands` | `5` | Number of isolate populations managed in the island model.
| `programs_per_island` | `null` | Optional cap on how many iterations are run on one island before rotating to the next. When `null`, `ProcessParallelController` approximates a fair value based on `max_iterations` and `num_islands`.
| `elite_selection_ratio` | `0.1` | Fraction of top programs guaranteed to survive selection.
| `exploration_ratio` | `0.2` | Probability of sampling diverse/novel programs when picking parents.
| `exploitation_ratio` | `0.7` | Probability of sampling from the current elites. Remaining probability (if any) is used for random exploration.
| `diversity_metric` | `"edit_distance"` | Currently always `edit_distance` internally; `feature_based` is reserved for future extensions.

### 4.3 MAP-Elites feature map

| Key | Default | Description |
| --- | --- | --- |
| `feature_dimensions` | `["complexity", "diversity"]` | Axes for the MAP-Elites grid. Built-ins: `complexity` (length), `diversity` (structure), `score`. Custom dimensions must match numeric metrics returned by your evaluator.
| `feature_bins` | `10` | Either a single integer or a mapping per dimension that defines how finely to discretize each axis.
| `diversity_reference_size` | `20` | Size of the rolling reference set used when computing structural diversity scores.

### 4.4 Island migration

| Key | Default | Description |
| --- | --- | --- |
| `migration_interval` | `50` | Number of generations between migrations.
| `migration_rate` | `0.1` | Fraction of top programs exchanged during migration events.

### 4.5 Novelty filtering & embeddings

| Key | Default | Description |
| --- | --- | --- |
| `embedding_model` | `null` | Name of the embedding model to call via `openevolve.embedding.EmbeddingClient`. Enables semantic deduplication.
| `embedding_api_base`, `embedding_api_key` | `null` | Endpoint and key for the embedding service. If unset, OpenEvolve reuses the main API credentials when possible.
| `similarity_threshold` | `0.99` | Cosine similarity gate applied to embeddings during `_is_novel`. Values close to 1 only reject nearly identical programs; decreasing it filters broader families. Set to `0` or a negative value to disable novelty rejection. When a candidate exceeds the threshold, OpenEvolve triggers an LLM “novelty judge” comparison before deciding whether to discard it.
| `novelty_llm` | auto | Automatically wired to the main LLM ensemble so the database can ask for textual novelty judgments after the embedding check.

This novelty system is what powers the `similarity_threshold`, `population_size`, etc. in the retrieval transformer example you cited.

### 4.6 Artifact storage
Evaluation artifacts are stored via `store_artifacts`. Artifacts smaller than `artifact_size_threshold` are serialized into the DB; larger blobs land under `artifacts_base_path/<program_id>/`. Retention is enforced when checkpoints run if `cleanup_old_artifacts` is `true`.

---

## 5. Evaluator (`evaluator` block)

### 5.1 Execution controls

| Key | Default | Description |
| --- | --- | --- |
| `timeout` | `300` | Maximum time (seconds) an `evaluate` call is allowed to run before being marked as a timeout.
| `max_retries` | `3` | Number of times a failed evaluation is retried before giving up.
| `parallel_evaluations` | `1` | Size of the evaluator’s async task pool (`TaskPool`). Setting this >1 allows concurrent test executions.
| `memory_limit_mb`, `cpu_limit` | `null` | Reserved for future resource sandboxing; not enforced yet.
| `distributed` | `false` | Placeholder flag for future distributed evaluators.

### 5.2 Cascade evaluation

| Key | Default | Behaviour |
| --- | --- | --- |
| `cascade_evaluation` | `true` | Enables the multi-stage evaluation pipeline. The evaluator looks for `evaluate_stage1/2/3` functions in your evaluation module and falls back to `evaluate` if they’re missing.
| `cascade_thresholds` | `[0.5, 0.75, 0.9]` | Minimum score required to advance from one stage to the next. Example configs often tighten this list (e.g., `[1.3]` to ensure a single threshold).

### 5.3 Artifact handling & feedback

| Key | Default | Description |
| --- | --- | --- |
| `enable_artifacts` | `true` | Controls whether evaluator artifacts are captured at all. Also respects the `ENABLE_ARTIFACTS` environment variable (defaults to `true`).
| `max_artifact_storage` | `100 MB` | Per-program quota for artifacts before old entries are evicted.
| `use_llm_feedback` | `false` | When `true`, prompts an LLM-based reviewer after each evaluation.
| `llm_feedback_weight` | `0.1` | Scale factor applied to each LLM feedback metric before combining it with numeric metrics.

Artifacts collected here are later surfaced to the prompt builder if `prompt.include_artifacts` is enabled.

---

## 6. Evolution Trace Logging (`evolution_trace` block)

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Turn on detailed iteration logging for research/RL datasets.
| `format` | `"jsonl"` | Output format: `jsonl`, `json`, or `hdf5`.
| `include_code` | `false` | Whether the raw parent/child code is written to the trace file.
| `include_prompts` | `true` | Adds prompt text and LLM responses to the trace.
| `output_path` | `null` | Custom file path. When unset, defaults to `<output_dir>/evolution_trace.<format>`.
| `buffer_size` | `10` | Number of trace entries accumulated before writing to disk.
| `compress` | `false` | For `jsonl` traces, write `.jsonl.gz` instead of plain text.

---

## 7. Legacy or example-only keys

- `allow_full_rewrites`: present in some historical example configs but ignored by the current loader. Use `diff_based_evolution: false` instead.
- Dataset-specific YAMLs under `examples/llm_prompt_optimization/` include extra sections (`dataset`, `tasks`, etc.) that are consumed by their bespoke scripts rather than by the core `Config` dataclasses. Refer to the README inside each example folder before reusing those fields.

---

## 8. Tips for crafting configs

1. Start from `configs/default_config.yaml` and only override fields you understand. The inline comments mirror the explanations above.
2. When running with local or vLLM backends, define both `llm.api_base` and `llm.api_key` (even if it is a dummy token) because most OpenAI-compatible servers still expect it.
3. If you enable `database.embedding_model`, also set `database.similarity_threshold` to reflect how aggressive you want novelty filtering to be. Setting it to `0.999` (as in `examples/retrieval_transformer/config_vllm.yaml`) only rejects near-duplicates; reducing it toward `0.95` increases exploration pressure.
4. Populate `prompt.system_message` with task-specific guidance. The README section “Crafting Effective System Messages” contains templates you can adapt.
5. Whenever you add new evaluator metrics, update `database.feature_dimensions` if you want the MAP-Elites grid to consider them. Otherwise they will still influence the fitness score via `combined_score`.

With this guide you can now locate any parameter—including the `population_size`, `archive_size`, `num_islands`, `elite_selection_ratio`, `exploitation_ratio`, and `similarity_threshold` in your retrieval transformer config—and understand exactly how OpenEvolve uses it.
