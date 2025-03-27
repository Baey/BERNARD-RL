# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: MIT


"""
Python module with the BERNARD robot model and configuration.
"""

# Register Gym environments.
from .bernard import BERNARD_CFG

__all__ = ["BERNARD_CFG"]
