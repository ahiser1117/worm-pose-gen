"""Reusable geometry, research models, and fail-closed worm-pose inference."""


def main() -> None:
    """Report the package's scientific status and direct users to explicit CLIs."""

    print(
        "worm-pose-gen: no deployment-authorized model; "
        "use scripts/infer_hdf5.py --allow-exploratory only for rejected-checkpoint diagnostics"
    )
