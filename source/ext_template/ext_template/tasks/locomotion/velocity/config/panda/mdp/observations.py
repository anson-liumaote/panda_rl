import torch
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

# Use string literal for type annotations to avoid circular imports
def motion_sequence_counter(env: Any, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_count=None) -> torch.Tensor:
    """Return the count of the motion sequence from 0 to max_count repeatedly.
    
    This function returns a counter for each environment instance that cycles from 0 to max_count.
    The counter is stored in the environment object and incremented on each call.
    
    Args:
        env: The environment.
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").
        
    Returns:
        A tensor containing the current motion sequence count for each environment.
        Shape is (num_envs, 1).
    """
    # Extract the asset to get the number of environments
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = asset.data.joint_pos.shape[0]
    
    # Create key for storing the counter in our custom context
    counter_key = "motion_sequence_counter"
    
    # Initialize the counter if it doesn't exist
    if not hasattr(env, "_motion_sequence_context"):
        env._motion_sequence_context = {}
    
    context = env._motion_sequence_context
    
    if counter_key not in context:
        context[counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.float32)
    
    # Get the current counter
    counters = context[counter_key]
    
    # Make sure the counters tensor has the correct size (in case num_envs changed)
    if counters.shape[0] != num_envs:
        counters = torch.zeros((num_envs, 1), device=env.device, dtype=torch.float32)
        context[counter_key] = counters
    
    # Store current counter values
    current_counters = counters.clone()
    
    # Increment the counters
    counters += 1
    context[counter_key] = torch.where(counters >= max_count, 
                                     torch.zeros_like(counters), 
                                     counters)
    
    # print(current_counters[0:3, 0])
    # Return the current counter values
    return current_counters