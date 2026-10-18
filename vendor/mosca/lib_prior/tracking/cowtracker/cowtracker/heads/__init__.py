# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CowTracker heads."""

from cowtracker.heads.tracking_head import CowTrackingHead
from cowtracker.heads.feature_extractor import FeatureExtractor
import cowtracker.thirdparty  # noqa: F401 - sets up vggt path
from vggt.heads.dpt_head import DPTHead

__all__ = ["CowTrackingHead", "FeatureExtractor", "DPTHead"]
