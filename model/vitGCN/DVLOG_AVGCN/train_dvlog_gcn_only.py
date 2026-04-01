"""Thin wrapper to launch vitGCN training with the DVLOG GCN-only profile."""

import os
import sys
import runpy
from pathlib import Path


if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    parent_dir = cur_dir.parent
    sys.path.insert(0, str(cur_dir))
    sys.path.insert(0, str(parent_dir))
    os.environ["VIT_GCN_PROFILE_MODULE"] = "config_dvlog_gcn_only"
    runpy.run_path(str(parent_dir / "vit_gcn_config_train.py"), run_name="__main__")
