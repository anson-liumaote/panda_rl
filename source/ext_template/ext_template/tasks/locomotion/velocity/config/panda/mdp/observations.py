# import torch
# from typing import TYPE_CHECKING, Any

# from isaaclab.assets import Articulation
# from isaaclab.managers import SceneEntityCfg

# # Use string literal for type annotations to avoid circular imports
# def motion_sequence_counter(env: Any, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_count: int = 83) -> torch.Tensor:
#     """Return the count of the motion sequence from 0 to max_count-1 repeatedly, normalized between 0 and 1.
    
#     This function returns a counter for each environment instance that cycles from 0 to max_count-1.
#     The raw counter is stored in the environment object and incremented on each call.
#     The returned values are normalized to range from 0.0 to 1.0.
    
#     Args:
#         env: The environment.
#         asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").
#         max_count: The maximum count value (exclusive). Counter goes from 0 to max_count-1. Defaults to 83.
        
#     Returns:
#         A tensor containing the normalized motion sequence count for each environment,
#         with values normalized between 0.0 and 1.0.
#         Shape is (num_envs, 1).
#     """
#     # Extract the asset to get the number of environments
#     asset: Articulation = env.scene[asset_cfg.name]
#     num_envs = asset.data.joint_pos.shape[0]
    
#     # Create key for storing the counter in our custom context
#     counter_key = "motion_sequence_counter"
#     raw_counter_key = "raw_motion_sequence_counter"
    
#     # Initialize the context and counters if they don't exist
#     if not hasattr(env, "_motion_sequence_context"):
#         env._motion_sequence_context = {}
    
#     context = env._motion_sequence_context
    
#     if raw_counter_key not in context:
#         context[raw_counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
    
#     if counter_key not in context:
#         context[counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.float32)
    
#     # Get the current raw counters
#     raw_counters = context[raw_counter_key]
    
#     # Make sure the counters tensor has the correct size (in case num_envs changed)
#     if raw_counters.shape[0] != num_envs:
#         raw_counters = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
#         context[raw_counter_key] = raw_counters
    
#     # Store current raw counter values
#     current_raw_counters = raw_counters.clone()
    
#     # Increment the raw counters
#     raw_counters += 1
    
#     # Reset counters that reach max_count back to 0
#     context[raw_counter_key] = torch.where(raw_counters >= max_count, 
#                                         torch.zeros_like(raw_counters), 
#                                         raw_counters)
    
#     # Normalize the counter values between 0 and 1
#     normalized_counters = current_raw_counters.float() / (max_count - 1)
#     context[counter_key] = normalized_counters
    
#     # For debugging (can be removed in production)
#     # print(f"Raw counts: {current_raw_counters[0:3, 0]}, Normalized: {normalized_counters[0:3, 0]}")
    
#     # Return the normalized counter values
#     return normalized_counters


# import torch
# from typing import TYPE_CHECKING, Any

# from isaaclab.assets import Articulation
# from isaaclab.managers import SceneEntityCfg

# # Use string literal for type annotations to avoid circular imports
# def motion_sequence_counter(env: Any, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_count: int = 83) -> torch.Tensor:
#     """Return the count of the motion sequence from 0 to max_count-1 repeatedly, normalized between 0 and 1.
    
#     This function returns a counter for each environment instance that cycles from 0 to max_count-1.
#     The raw counter is stored in the environment object and incremented on each call.
#     The returned values are normalized to range from 0.0 to 1.0.
    
#     Args:
#         env: The environment.
#         asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").
#         max_count: The maximum count value (exclusive). Counter goes from 0 to max_count-1. Defaults to 83.
        
#     Returns:
#         A tensor containing the normalized motion sequence count for each environment,
#         with values normalized between 0.0 and 1.0.
#         Shape is (num_envs, 1).
#     """
#     # Extract the asset to get the number of environments
#     asset: Articulation = env.scene[asset_cfg.name]
#     num_envs = asset.data.joint_pos.shape[0]
    
#     # Create key for storing the counter in our custom context
#     counter_key = "motion_sequence_counter"
#     raw_counter_key = "raw_motion_sequence_counter"
    
#     # Initialize the context and counters if they don't exist
#     if not hasattr(env, "_motion_sequence_context"):
#         env._motion_sequence_context = {}
    
#     context = env._motion_sequence_context
    
#     if raw_counter_key not in context:
#         context[raw_counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
    
#     if counter_key not in context:
#         context[counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.float32)
    
#     # Get the current raw counters
#     raw_counters = context[raw_counter_key]
    
#     # Make sure the counters tensor has the correct size (in case num_envs changed)
#     if raw_counters.shape[0] != num_envs:
#         raw_counters = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
#         context[raw_counter_key] = raw_counters
    
#     # Store current raw counter values
#     current_raw_counters = raw_counters.clone()
    
#     # Increment the raw counters
#     raw_counters += 1
    
#     # Reset counters that reach max_count back to 0
#     context[raw_counter_key] = torch.where(raw_counters >= max_count, 
#                                         torch.zeros_like(raw_counters), 
#                                         raw_counters)
    
#     # Reset counters for terminated environments
#     if hasattr(env, "reset_buf"):
#         reset_mask = env.reset_buf.view(-1, 1).to(dtype=torch.bool)
#         if reset_mask.any():
#             context[raw_counter_key] = torch.where(reset_mask, 
#                                                torch.zeros_like(raw_counters),
#                                                context[raw_counter_key])
            
#             # Update current_raw_counters for reset environments
#             current_raw_counters = torch.where(reset_mask,
#                                              torch.zeros_like(current_raw_counters),
#                                              current_raw_counters)
    
#     # Normalize the counter values between 0 and 1
#     normalized_counters = current_raw_counters.float() / (max_count - 1)
#     context[counter_key] = normalized_counters
    
#     # For debugging (can be removed in production)
#     print(f"Raw counts: {current_raw_counters[0:3, 0]}, Normalized: {normalized_counters[0:3, 0]}")
    
#     # Return the normalized counter values
#     return normalized_counters


import torch
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

# Use string literal for type annotations to avoid circular imports
def motion_sequence_counter(env: Any, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                           max_count: int = 83, loop_start: int = 0) -> torch.Tensor:
    """Return the count of the motion sequence with customizable looping behavior, normalized between 0 and 1.
    
    This function returns a counter for each environment instance that has two phases:
    1. Initial phase: Counter goes from 0 to loop_start-1 (skipped if loop_start is 0)
    2. Loop phase: Counter repeatedly cycles from loop_start to max_count-1
    
    The raw counter is stored in the environment object and incremented on each call.
    The returned values are normalized to range from 0.0 to 1.0.
    
    Args:
        env: The environment.
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").
        max_count: The maximum count value (exclusive). Counter goes from 0 to max_count-1. Defaults to 83.
        loop_start: The value where looping begins. After reaching max_count-1, counter goes back to this value.
                   If set to 0, the entire range is looped. Defaults to 0.
        
    Returns:
        A tensor containing the normalized motion sequence count for each environment,
        with values normalized between 0.0 and 1.0.
        Shape is (num_envs, 1).
    """
    # Validate input parameters
    assert 0 <= loop_start < max_count, f"loop_start must be between 0 and max_count-1, got {loop_start}"
    
    # Extract the asset to get the number of environments
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = asset.data.joint_pos.shape[0]
    
    # Create keys for storing the counter in our custom context
    counter_key = "motion_sequence_counter"
    raw_counter_key = "raw_motion_sequence_counter"
    phase_key = "motion_sequence_phase"  # 0 for initial phase, 1 for loop phase
    
    # Initialize the context and counters if they don't exist
    if not hasattr(env, "_motion_sequence_context"):
        env._motion_sequence_context = {}
    
    context = env._motion_sequence_context
    
    if raw_counter_key not in context:
        context[raw_counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
    
    if counter_key not in context:
        context[counter_key] = torch.zeros((num_envs, 1), device=env.device, dtype=torch.float32)
    
    if phase_key not in context:
        # Start in phase 0 if loop_start > 0, otherwise start in phase 1
        initial_phase = 0 if loop_start > 0 else 1
        context[phase_key] = torch.full((num_envs, 1), initial_phase, device=env.device, dtype=torch.long)
    
    # Get the current raw counters and phases
    raw_counters = context[raw_counter_key]
    phases = context[phase_key]
    
    # Make sure the tensors have the correct size (in case num_envs changed)
    if raw_counters.shape[0] != num_envs:
        raw_counters = torch.zeros((num_envs, 1), device=env.device, dtype=torch.long)
        context[raw_counter_key] = raw_counters
        initial_phase = 0 if loop_start > 0 else 1
        phases = torch.full((num_envs, 1), initial_phase, device=env.device, dtype=torch.long)
        context[phase_key] = phases
    
    # Store current raw counter values before updating
    current_raw_counters = raw_counters.clone()
    
    # Increment the raw counters
    raw_counters += 1
    
    # Handle phase transitions and counter reset logic
    # Phase 0: Initial sequence (0 to loop_start-1)
    # Phase 1: Loop sequence (loop_start to max_count-1)
    
    # Check if any counters in phase 0 have reached loop_start
    phase0_transition_mask = (phases == 0) & (raw_counters >= loop_start)
    if phase0_transition_mask.any():
        # Transition from phase 0 to phase 1
        phases = torch.where(phase0_transition_mask, 
                           torch.ones_like(phases),
                           phases)
        context[phase_key] = phases
    
    # Check if any counters in phase 1 have reached max_count
    phase1_reset_mask = (phases == 1) & (raw_counters >= max_count)
    if phase1_reset_mask.any():
        # Reset counters in phase 1 back to loop_start (not to 0)
        raw_counters = torch.where(phase1_reset_mask,
                                  torch.full_like(raw_counters, loop_start),
                                  raw_counters)
        context[raw_counter_key] = raw_counters
    
    # Reset counters for terminated environments
    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.view(-1, 1).to(dtype=torch.bool)
        if reset_mask.any():
            # Reset counter to 0 and phase to initial
            initial_phase = 0 if loop_start > 0 else 1
            context[raw_counter_key] = torch.where(reset_mask, 
                                               torch.zeros_like(raw_counters),
                                               context[raw_counter_key])
            
            context[phase_key] = torch.where(reset_mask,
                                          torch.full_like(phases, initial_phase),
                                          phases)
            
            # Update current_raw_counters for reset environments
            current_raw_counters = torch.where(reset_mask,
                                             torch.zeros_like(current_raw_counters),
                                             current_raw_counters)
    
    # Normalize the counter values between 0 and 1
    normalized_counters = current_raw_counters.float() / (max_count - 1)
    context[counter_key] = normalized_counters
    
    # For debugging (can be removed in production)
    # if num_envs > 0:
    #     print(f"Raw counts: {current_raw_counters[0:3, 0]}, Phase: {phases[0:3, 0]}, Normalized: {normalized_counters[0:3, 0]}")
    
    # Return the normalized counter values
    return normalized_counters