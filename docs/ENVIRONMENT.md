# Environment

The project uses Python 3.13 and `uv` exclusively. The virtual environment is
intentionally stored at `.venv` in the repository and is ignored by Git.
Uv-managed interpreter installations use `.uv-python`; disposable package and
application caches are stored beneath `.cache`. All three paths are ignored by
Git.

Checkpoints, generated datasets, profiler traces, and experiment outputs that
can accumulate belong beneath the canonical external root:

```text
/temp_data4/alex/external_artifacts
```

On this machine that path resolves to
`/storage/fs/temp_data4/alex/external_artifacts`. Configuration and user-facing
commands should continue to use the canonical `/temp_data4` path.

## Command wrapper

Run every project `uv` command through the checked-in wrapper so separate shell
invocations receive the same paths and CUDA visibility. The wrapper sets the uv
project root explicitly, so it also works when invoked from another directory:

```bash
scripts/project_env.sh uv run python scripts/preflight.py
```

The wrapper also sends Matplotlib, PyTorch, and XDG caches to writable local
paths and forces Matplotlib's non-interactive `Agg` backend.

## Bootstrap

After changing `pyproject.toml` or on a new checkout:

```bash
scripts/project_env.sh uv lock
scripts/bootstrap_environment.sh
```

The bootstrap creates the external output directories, performs a write/read
test, synchronizes from `uv.lock`, and validates the Python environment. CUDA
access can require a sandbox approval; request it early with:

```bash
scripts/bootstrap_environment.sh --require-cuda
```

After dependencies have been cached, reproducibility can be checked without
network access:

```bash
UV_OFFLINE=1 scripts/project_env.sh uv sync --frozen --python 3.13
```

## Required runtime invariants

- Python is `>=3.13,<3.14` and `.python-version` pins minor version `3.13`.
- `CUDA_VISIBLE_DEVICES=0`; physical device 0 appears as logical `cuda:0`.
- PyTorch must see exactly one CUDA device for GPU-dependent work.
- Source HDF5 files are read-only inputs and are never overwritten.
- Lightning logs, checkpoints, and profiler outputs must use explicit paths
  beneath the external root rather than framework defaults.

## Observed Phase 0 environment

Recorded on 2026-08-18 after running
`scripts/bootstrap_environment.sh --require-cuda`:

- Python: 3.13.14 at `.venv/bin/python3`
- PyTorch: 2.13.0+cu130; CUDA runtime: 13.0
- Lightning: 2.6.5
- h5py: 3.16.0; HDF5: 2.0.0
- NumPy: 2.5.2; Matplotlib: 3.11.1
- physical CUDA device: index 0, UUID
  `GPU-f72d2ba7-8334-183e-e368-2c527e8a39e6`, PCI `00000000:01:00.0`
- visible logical CUDA device: index 0; PyTorch device count: exactly 1
- GPU: NVIDIA RTX 6000 Ada Generation, 49,140 MiB, driver 610.43.02
- configured external root: `/temp_data4/alex/external_artifacts`
- resolved external root: `/storage/fs/temp_data4/alex/external_artifacts`
- source data mount: read-only NFSv4 at `/storage/fs/store1`
- external artifact mount: writable ZFS at `/storage/fs/temp_data4`

The active sandbox required a narrow approved host execution for CUDA identity
and bootstrap validation. Dependency resolution is performed only by the
checked-in wrapper; no system packages, drivers, or system Python are changed.

## Deterministic seed policy

The canonical base seed is 20260818. Split generation, NumPy, PyTorch, Python
`random`, augmentations, and data-loader workers derive named seeds from that
base seed. Lightning deterministic mode is enabled for correctness and smoke
tests. Performance experiments may retain documented nondeterministic CUDA
kernels when deterministic alternatives materially reduce throughput; those
runs must record the flag and are repeated when variability could change a
decision. Model-selection experiments use at least three seeds when the result
lies within the predeclared practical-effect margin; otherwise one seed plus a
bootstrap interval over held-out samples is sufficient for elimination.

## Source-recording access

The `nir_videos/*.h5` paths are symlinks into the accepted read-only NFS mount.
Readers open source files with mode `r`, use bounded frame-indexed reads, and
never copy or modify the recordings. Concurrent HDF5 readers are capped in
`configs/resource_budget.yaml`. The current NFS configuration is accepted as
provided and is not a project tuning target.
