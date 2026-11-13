#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D69 owned training wrapper: route the app's gated-delta call, then exec the
qualified lora-smoke two-host entry point unchanged.

Usage (inside the qualified worker, via the guardian):
    python3 d69_train_wrapper.py --config <app-config.json> [app args...]

Environment knobs (all explicit, no defaults changed in the app):
    BC250_D69_ROUTE=0      run the pure 5.22.1 reference (no routing)
    BC250_D69_NATIVE       installed | fused (default fused)
    BC250_D69_CANDIDATE    staged D68 candidate module path
                           (default: <this dir>/bc250_gated_delta.py)

The wrapper verifies the app config's model_type is qwen3_5_text before
installing the module-scope route, forwards argv verbatim to
train_two_host.py, and always uninstalls exactly, writing
route-record-rank<N>.json into the app output directory. Application source,
entry points, outputs and checkpoints are never modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_qwen35_gated_delta_d69 as routing  # noqa: E402

PROJECT = Path("/home/nik/bc250-projects/lora-smoke/project")
ENTRY = PROJECT / "train_two_host.py"


def extract_config(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    raise ValueError("wrapper requires --config <app-config.json>")


def main() -> int:
    config_path = extract_config(sys.argv[1:])
    app_config = json.loads(config_path.read_text())
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(app_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / f"route-record-rank{rank}.json"

    route_enabled = os.environ.get("BC250_D69_ROUTE", "1") != "0"
    target = os.environ.get("BC250_D69_NATIVE", "fused")
    candidate = os.environ.get(
        "BC250_D69_CANDIDATE", str(HERE / "bc250_gated_delta.py")
    )
    record = None
    summary = {
        "fixture": "d69_train_wrapper",
        "route_enabled": route_enabled,
        "target": target if route_enabled else "reference",
        "rank": rank,
        "config": str(config_path),
    }
    try:
        if route_enabled:
            model_path = PROJECT / app_config["model"]
            record = routing.install_for_app(
                model_path, target=target, candidate_path=candidate
            )
            summary.update(
                {
                    "native_module": record["native_module"],
                    "fused": record["fused"],
                    "model_type_verified": True,
                }
            )
            print(
                f"D69_ROUTE installed target={target} fused={record['fused']} "
                f"model_type verified",
                flush=True,
            )
        else:
            print("D69_ROUTE disabled (reference run)", flush=True)
        # Direct script execution puts the script's directory on sys.path;
        # runpy does not. The app entry point imports its sibling support
        # module (bc250_lora_support) from PROJECT, so replicate that.
        sys.path.insert(0, str(PROJECT))
        sys.argv = [str(ENTRY), *sys.argv[1:]]
        runpy.run_path(str(ENTRY), run_name="__main__")
        summary["entry_exit"] = 0
        return 0
    except SystemExit as exc:
        summary["entry_exit"] = exc.code
        raise
    finally:
        if record is not None:
            summary["native_calls"] = record["native_calls"]
            summary["fallbacks"] = record["fallbacks"]
            routing.uninstall(record)
            summary["uninstalled"] = True
            print(
                f"D69_ROUTE uninstalled native_calls={record['native_calls']} "
                f"fallbacks={record['fallbacks']}",
                flush=True,
            )
        record_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
