import torch
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

# Use string literal for type annotations to avoid circular imports
def motion_sequence_counter(env: Any, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_count: int = 83) -> torch.Tensor:
    """Return the count of the motion sequence from 0 to max_count-1 repeatedly, normalized between 0 and 1.
    
    This function returns a counter for each environment instance that cycles from 0 to max_count-1.
    The raw counter is stored in the environment object and incremented on each call.
    The returned values are normalized to range from 0.0 to 1.0.
    
    Args:
        env: The environment.
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").
        max_count: The maximum count value (exclusive). Counter goes from 0 to max_count-1. Defaults to 83.
        
    Returns:
        A tensor containing the normalized motion sequence count for each environment,
        with values normalized between 0.0 and 1.0.
        Shape is (num_envs, 1).
    """
    # Extract the asset to get the number of environments
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = asset.data.joint_pos.shape[0]
    
    # Create key for storing the counter in our custom context
    counter_key = "motion_sequence_counter"
    raw_counter_key = "raw_motion_sequence_counter"
    
    # Initialize the context and counters if they don't exist
    if not hasattr(env, "_motion_sequence_context"):
        env._motion_sequence_context = {}
    
    context = env._motion_sequence_context
    
    if raw_counter_key not in context:
        context[raw_counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
    
    if counter_key not in context:
        context[counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.float32)
    
    # Get the current raw counters
    raw_counters = context[raw_counter_key]
    
    # Make sure the counters tensor has the correct size (in case num_envs changed)
    if raw_counters.shape[0] != num_envs:
        raw_counters = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
        context[raw_counter_key] = raw_counters
    
    # Store current raw counter values
    current_raw_counters = raw_counters.clone()
    
    # Increment the raw counters
    raw_counters += 1
    
    # Reset counters that reach max_count back to 0
    context[raw_counter_key] = torch.where(raw_counters >= max_count, 
                                        torch.zeros_like(raw_counters), 
                                        raw_counters)
    
    # Normalize the counter values between 0 and 1
    normalized_counters = current_raw_counters.float() / (max_count - 1)
    context[counter_key] = normalized_counters
    
    # For debugging (can be removed in production)
    # print(f"Raw counts: {current_raw_counters[0:3, 0]}, Normalized: {normalized_counters[0:3, 0]}")
    
    # Return the normalized counter values
    return normalized_counters