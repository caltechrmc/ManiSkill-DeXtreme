"""
Implements Automatic Domain Randomization (ADR) as detailed in
the DeXtreme paper.
"""
from enum import Enum
from dataclasses import dataclass

class Distribution(Enum):
    UNIFORM = 0
    LOGUNIFORM = 1
    GAUSSIAN = 2
    NORMAL = 3

class NoiseType(Enum):
    SCALING = 0
    ADDITIVE = 1
    SET_VALUE = 2

@dataclass
class DomainParameter:
    name: str
    type: NoiseType
    dist: Distribution
    minv: float
    maxv: float

# Edit this list to add ADR parameters. (DeXtreme Table 3)
ADR_PARAMETERS = [
    # Hand
    DomainParameter("hand.mass", NoiseType.SCALING, Distribution.UNIFORM, 0.4, 1.5),
    DomainParameter("hand.scale", NoiseType.SCALING, Distribution.UNIFORM, 0.95, 1.05),
    DomainParameter("hand.friction", NoiseType.SCALING, Distribution.UNIFORM, 0.8, 1.2),
    DomainParameter("hand.armature", NoiseType.SCALING, Distribution.UNIFORM, 0.8, 1.02),
    DomainParameter("hand.effort", NoiseType.SCALING, Distribution.UNIFORM, 0.9, 1.1),
    DomainParameter("hand.joint_stiffness", NoiseType.SCALING, Distribution.LOGUNIFORM, 0.3, 3.0),
    DomainParameter("hand.joint_damping", NoiseType.SCALING, Distribution.LOGUNIFORM, 0.75, 1.5),
    DomainParameter("hand.restitution", NoiseType.ADDITIVE, Distribution.UNIFORM, 0.0, 0.4),
    
    # Object
    DomainParameter("object.mass", NoiseType.SCALING, Distribution.UNIFORM, 0.4, 1.6),
    DomainParameter("object.friction", NoiseType.SCALING, Distribution.UNIFORM, 0.3, 0.9),
    DomainParameter("object.scale", NoiseType.SCALING, Distribution.UNIFORM, 0.95, 1.05),
    # TODO: External Forces
    DomainParameter("object.restitution", NoiseType.ADDITIVE, Distribution.UNIFORM, 0.0, 0.4),
    DomainParameter("observation.obj_pose_delay_prob", NoiseType.SET_VALUE, Distribution.UNIFORM, 0.0, 0.05),
    DomainParameter("observation.obj_pose_freq", NoiseType.SET_VALUE, Distribution.UNIFORM, 1.0, 1.0),
    DomainParameter("observation.obs_corr_noise", NoiseType.ADDITIVE, Distribution.GAUSSIAN, 0.0, 0.04),
    DomainParameter("observation.obs_uncorr_noise", NoiseType.ADDITIVE, Distribution.GAUSSIAN, 0.0, 0.04),
    DomainParameter("observation.random_pose_injection", NoiseType.SET_VALUE, Distribution.UNIFORM, 0.3, 0.3),
    
    # Action
    DomainParameter("action.action_delay_prob", NoiseType.SET_VALUE, Distribution.UNIFORM, 0.0, 0.05),
    DomainParameter("action.action_latency", NoiseType.SET_VALUE, Distribution.UNIFORM, 0.0, 0.0),
    DomainParameter("action.action_corr_noise", NoiseType.ADDITIVE, Distribution.GAUSSIAN, 0.0, 0.04),
    DomainParameter("action.action_uncorr_noise", NoiseType.ADDITIVE, Distribution.GAUSSIAN, 0.0, 0.04),
    DomainParameter("action.rna_alpha", NoiseType.SET_VALUE, Distribution.UNIFORM, 0.0, 0.0),
    
    # Environment
    DomainParameter("environment.gravity", NoiseType.ADDITIVE, Distribution.NORMAL, 0.0, 0.5),
]

ADR_PARAMETER_DICT = {param.name: param for param in ADR_PARAMETERS}

def get_param(name: str) -> DomainParameter:
    return ADR_PARAMETER_DICT[name]