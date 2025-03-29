from dataclasses import dataclass

@dataclass
class ReorientCubeEnvRewardConfig:
    # Scaling factor for reward term that brings the cube orientation close to the goal
    orientation_gain: float = 1.0

    # Additive constant in denominator of position reward term
    orientation_eps: float = 0.1
    
    # Scaling factor for reward term that brings the cube position close to the goal
    position_gain: float = -10.0
    
    # Scaling factor for reward term that prevents raw actions from being too large
    action_penalty: float = -0.001
    
    # Scaling factor for reward term that prevents rapid changes in joint targets
    action_delta_penalty: float = -0.25
    
    # Scaling factor for reward term that penalizes fingers from moving too quickly
    velocity_penalty: float = -0.003
    
    # A large reward for getting the cube to the target
    success_bonus: float = 250.0
    
    # A penalty for dropping the cube
    drop_penalty: float = -250.0 # TODO: This value is not from the paper
    
    # A penalty for not reaching a goal in time
    timeout_penalty: float = drop_penalty * 0.5

@dataclass
class ReorientCubeEnvConfig:
    # A configuration for the shaped reward function
    reward: ReorientCubeEnvRewardConfig
    
    # The number of DOFs in the hand
    dofs: int = 16
    
    # Whether to use Automatic Domain Randomization (ADR)
    # TODO: Not yet implemented
    use_adr: bool = False
    
    # The maximum amount of time to reach a goal that we allow (in seconds)
    stuck_timeout: int = 80
    
    # The threshold to consider a target orientation reached (rad)
    success_tolerance: float = 0.1
    
    # The number of steps for which a goal must be held before it is considered a success
    success_hold_duration: int = 0
    
    # If the cube is beyond this distance from the goal, we consider it to have been dropped
    drop_distance: float = 0.5
    
    # Maximum number of consecutive goals reached before we just terminate the episode
    max_consecutive_successes: int = 0