"""Compatibility entrypoint.

Use vit_gcn_config_train.py for the refactored orchestrator and
vit_gcn_runtime_utils.py for runtime helpers.
"""

import runpy


if __name__ == '__main__':
    runpy.run_module('vit_gcn_config_train', run_name='__main__')
