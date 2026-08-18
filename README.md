# Worm Pose Gen

Research code for probabilistic 2D *C. elegans* centerline and body-angle
estimation from NIR behavior video. The active research specification is
[`worm_pose_agent_orchestrator.md`](worm_pose_agent_orchestrator.md).

## Environment

The project requires Python 3.13 and uses `uv` for all Python and dependency
operations. Bootstrap and validate the existing repository-local `.venv` with:

```bash
scripts/project_env.sh uv lock
scripts/bootstrap_environment.sh
```

Run project Python commands through the environment wrapper:

```bash
scripts/project_env.sh uv run python scripts/preflight.py
```

See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for storage, CUDA, and
reproducibility details. Training and inference commands will be documented as
their implementations are established by controlled experiments.
