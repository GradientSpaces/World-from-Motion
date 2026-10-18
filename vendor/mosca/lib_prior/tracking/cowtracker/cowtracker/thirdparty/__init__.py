# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Third-party modules.

This module sets up sys.path for third-party packages.
"""

import os
import sys

# Add vggt to sys.path so that 'from vggt.xxx' imports work
_vggt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt")
if _vggt_path not in sys.path:
    sys.path.insert(0, _vggt_path)

