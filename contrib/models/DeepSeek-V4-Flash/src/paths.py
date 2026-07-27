# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for DeepSeek-V4-Flash checkpoint locations.

Every script in this port needs the same three paths, and they were
previously hardcoded to one developer's NVMe mount. Resolve them here from
`DSV4_MODEL_PATH` so a different box only has to set one env var:

    export DSV4_MODEL_PATH=/mnt/data/models/DeepSeek-V4-Flash

The checkpoint's own `inference/` directory is added to sys.path by
`add_hf_inference_to_syspath()` — that's where HF's `model.py` lives, and it
must be importable as a top-level `model` module because it does
`from kernel import ...` internally.
"""

import os
import sys


def model_path() -> str:
    """Root of the HF checkpoint download (contains model-*.safetensors)."""
    return os.environ.get("DSV4_MODEL_PATH", "/mnt/data/models/DeepSeek-V4-Flash")


def hf_inference_dir() -> str:
    """The checkpoint's `inference/` dir, holding HF's model.py + config.json."""
    return os.environ.get(
        "DSV4_HF_INFERENCE_DIR", os.path.join(model_path(), "inference")
    )


def config_json() -> str:
    """The inference-flavoured config (ModelArgs field names, not HF's)."""
    return os.environ.get(
        "DSV4_CONFIG", os.path.join(hf_inference_dir(), "config.json")
    )


def add_hf_inference_to_syspath() -> str:
    """Put `inference/` on sys.path so `import model` resolves to HF's."""
    d = hf_inference_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    return d
