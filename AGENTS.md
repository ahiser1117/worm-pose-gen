# Agent instructions

How to run, test and measure in this repository without waiting on it. A
review of an Oct 2026 workflow found that agents spent 75-90% of their time
waiting on unit tests and GPU evaluations and 5-15 minutes in the model; the
rules below target that time.

## Environment

- Run Python through the wrapper: `scripts/project_env.sh uv run --no-sync --frozen python ...`.
- In a git worktree, test and run the worktree's code with the main venv:
  `env PYTHONPATH=<worktree>/src /home/alex/git/worm-pose-gen/scripts/project_env.sh uv run --no-sync --frozen python <worktree>/scripts/...`.
  Check once that `worm_pose_gen.__file__` resolves inside the worktree.
- This machine is flv-c4: a shared host, often with dozens of users. Never use
  its GPUs. GPU runs go over ssh to flv-c3 first, then flv-c2; `nvidia-smi`
  there and pick an idle GPU. Home and `/temp_data4` are the same NFS paths on
  every host. The remote login shell is fish (`; and`, wrap loops in
  `bash -c`). `project_env.sh` forces `CUDA_VISIBLE_DEVICES=0`, so set the
  device after it: `... project_env.sh env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> uv run ...`.
- Run outputs go under `/temp_data4/alex/external_artifacts/`, not in the repo.

## Unit tests

- Use `scripts/run_tests.py` (README, Tests). It runs one process per test
  class in parallel, caps OpenMP/MKL threads, and tests the checkout it sits
  in, so call `<worktree>/scripts/run_tests.py` from a worktree.
- While iterating, run the modules you touch, or `--fast` (about a minute);
  `--fast` skips the `@slow` tests. Run the whole suite (about 7 minutes) once,
  before committing, not after every edit.
- Never start Python tests without a thread cap. Torch takes 128 threads per
  process here; uncapped runs from parallel agents made `tests.test_pipeline`
  take 30 minutes instead of 7. When calling `unittest` directly, set
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg`.
- Mark a new test `@slow` (`from tests.slow import slow`) when it takes ten
  seconds or more. Prefer small fixtures and short `stage_steps` for tests
  that check plumbing rather than fit quality.
- Browser tests (`tests/browser/*.cjs`) take minutes each; run only the page
  you changed, and only after the Python side passes.

## Evaluations

- `scripts/evaluate_sequence_set.py` on the full seven-clip set takes 8-11
  minutes on one GPU. Iterate on one or two clips (`--clip <name>`) and run
  the full set once for the final numbers.
- Compute baseline numbers once and reuse them. Put them under
  `/temp_data4/alex/external_artifacts/baselines/<base commit>_<checkpoint>_<preset>/`
  and check for an existing result before running one. In a multi-agent
  workflow, a setup step computes the baseline before the agents fan out;
  each agent does not rerun it.
- Compare before and after on the same GPU host with the same checkpoint and
  preset.

## Waiting on long runs

- Start anything over a minute (test suites, evaluations, ssh GPU jobs) in the
  background and wait for its completion notification. Do not block in
  foreground `until ...; sleep` loops: they hit the 10-minute Bash limit and
  turn one wait into several turns.
- Have long runs write a log with a clear final line (`OK`/`FAILED`, `done`)
  and read only its tail.
- Keep parallel CPU work bounded: several agents each running the full suite
  at once on flv-c4 slow every one of them down. Put heavy CPU runs on
  flv-c3/flv-c2 when flv-c4 is loaded (`uptime`).
