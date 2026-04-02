"""Thin wrapper to launch vitGCN training with the DVLOG profile."""

import os
import sys
import argparse
import runpy
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DVLOG experiments with a single switch")
    parser.add_argument(
        "--exp",
        default=os.getenv("DVLOG_EXPERIMENT", "fusion"),
        choices=["fusion", "av_only_new", "av_only_legacy", "gcn_only"],
        help="Experiment mode",
    )
    args = parser.parse_args()

    cur_dir = Path(__file__).resolve().parent
    parent_dir = cur_dir.parent
    sys.path.insert(0, str(cur_dir))
    sys.path.insert(0, str(parent_dir))
    os.environ["DVLOG_EXPERIMENT"] = args.exp
    os.environ["VIT_GCN_PROFILE_MODULE"] = "config_dvlog"
    runpy.run_path(str(parent_dir / "vit_gcn_config_train.py"), run_name="__main__")

