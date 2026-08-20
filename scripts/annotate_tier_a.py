#!/usr/bin/env python3
"""Launch the local browser-based Tier-A annotation tool."""

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

from worm_pose_gen.annotation_tool import (
    AnnotationProtocol,
    AnnotationSession,
    create_server,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-id", required=True, help="Stable pseudonym stored in every label")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "experiments/scientific_exp_001_annotation/selection_manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="New or resumable JSON session file")
    parser.add_argument("--primary-count", type=int, default=30)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--repeat-delay-hours", type=float, default=168.0)
    parser.add_argument("--selection-seed", type=int, default=20260819)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the tool in the default browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = AnnotationProtocol(
        primary_count=args.primary_count,
        repeat_count=args.repeat_count,
        repeat_delay_hours=args.repeat_delay_hours,
        selection_seed=args.selection_seed,
    )
    session = AnnotationSession(args.manifest, args.output, args.annotator_id, protocol)
    server = create_server(session, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"Tier-A annotation tool: {url}")
    print(f"Session file: {session.output_path}")
    print("Press Ctrl-C to stop; saved annotations are already durable.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.frame_repository.close()


if __name__ == "__main__":
    main()
