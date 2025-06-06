# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def feet_contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """"""
    # extract the used quantities (to enable type-hinting)
    feet_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = torch.norm(feet_contact_sensor.data.net_forces_w_history, dim=-1)
    # compute the violation
    return net_contact_forces[:, 0, sensor_cfg.body_ids]
