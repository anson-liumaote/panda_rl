import numpy as np
from scipy.interpolate import interp1d

def resample_joint_poses(input_file, output_file, velocity_output_file=None, target_freq=50, blend_range=5):
    """
    Resample joint pose data to a target frequency using linear interpolation.
    Handles cyclic motion by blending start and end poses.
    Reorders joints from [fr(γαβ), fl(γαβ), hr(γαβ), hl(γαβ)] to
    [fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf]
    
    Parameters:
    input_file (str): Path to input file containing joint poses with timestamps
    output_file (str): Path to save resampled joint poses
    velocity_output_file (str, optional): Path to save joint velocities
    target_freq (float): Target frequency in Hz (default: 50)
    blend_range (int): Number of frames to blend at the start/end (default: 5)
    """
    # Load the data
    data = np.loadtxt(input_file)
    
    # Extract timestamps from the last column
    timestamps = data[:, -1]
    joint_data = data[:, :-1]
    
    # # Create cyclic data by repeating and blending
    # # Add last few frames to start and first few frames to end
    # joint_data_extended = np.vstack([
    #     joint_data[-blend_range:],  # Add last frames to start
    #     joint_data,
    #     joint_data[:blend_range]    # Add first frames to end
    # ])
    
    # # Create extended timestamps
    # time_step = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    # timestamps_extended = np.concatenate([
    #     timestamps[0] - np.arange(blend_range, 0, -1) * time_step,
    #     timestamps,
    #     timestamps[-1] + np.arange(1, blend_range + 1) * time_step
    # ])
    
    # Calculate the duration and create new timestamps at target frequency
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_freq) + 1
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    
    # Create interpolation function for each joint using linear interpolation
    interpolators = [interp1d(timestamps, joint_data[:, i], kind='linear') 
                    for i in range(joint_data.shape[1])]
    
    # Generate resampled data
    resampled_joints = np.zeros((len(new_timestamps), joint_data.shape[1]))
    for i, interpolator in enumerate(interpolators):
        resampled_joints[:, i] = interpolator(new_timestamps)
    
    # Reorder the joints
    # Original order: fr[gamma,alpha,beta] fl[gamma,alpha,beta] hr[gamma,alpha,beta] hl[gamma,alpha,beta]
    # New order: fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf
    
    reordered_joints = np.zeros_like(resampled_joints)
    
    # Mapping from old to new indices:
    # fl_hip (original fl_gamma: index 3) -> index 0
    reordered_joints[:, 0] = resampled_joints[:, 3]
    
    # fr_hip (original fr_gamma: index 0) -> index 1
    reordered_joints[:, 1] = resampled_joints[:, 0]
    
    # rl_hip (original hl_gamma: index 9) -> index 2
    reordered_joints[:, 2] = resampled_joints[:, 9]
    
    # rr_hip (original hr_gamma: index 6) -> index 3
    reordered_joints[:, 3] = resampled_joints[:, 6]
    
    # fl_thigh (original fl_alpha: index 4) -> index 4
    reordered_joints[:, 4] = resampled_joints[:, 4]
    
    # fr_thigh (original fr_alpha: index 1) -> index 5
    reordered_joints[:, 5] = resampled_joints[:, 1]
    
    # rl_thigh (original hl_alpha: index 10) -> index 6
    reordered_joints[:, 6] = resampled_joints[:, 10]
    
    # rr_thigh (original hr_alpha: index 7) -> index 7
    reordered_joints[:, 7] = resampled_joints[:, 7]
    
    # fl_calf (original fl_beta: index 5) -> index 8
    reordered_joints[:, 8] = resampled_joints[:, 5]
    
    # fr_calf (original fr_beta: index 2) -> index 9
    reordered_joints[:, 9] = resampled_joints[:, 2]
    
    # rl_calf (original hl_beta: index 11) -> index 10
    reordered_joints[:, 10] = resampled_joints[:, 11]
    
    # rr_calf (original hr_beta: index 8) -> index 11
    reordered_joints[:, 11] = resampled_joints[:, 8]
    
    # Save the resampled and reordered data
    np.savetxt(output_file, reordered_joints, fmt='%.6f', delimiter=' ')
    
    # Calculate joint velocities if requested
    if velocity_output_file:
        # Calculate time difference between frames (should be constant for resampled data)
        dt = 1.0 / target_freq
        
        # Calculate velocities (rad/s) using central difference
        velocities = np.zeros_like(reordered_joints)
        
        # Central difference for middle frames
        velocities[1:-1] = (reordered_joints[2:] - reordered_joints[:-2]) / (2 * dt)
        
        # For first frame, use last and second frames (cyclic boundary)
        velocities[0] = (reordered_joints[1] - reordered_joints[-1]) / (2 * dt)
        
        # For last frame, use last-1 and first frames (cyclic boundary)
        velocities[-1] = (reordered_joints[0] - reordered_joints[-2]) / (2 * dt)
        
        # Save velocities to file
        np.savetxt(velocity_output_file, velocities, fmt='%.6f', delimiter=' ')
        print(f"Joint velocities saved to {velocity_output_file}")
    
    return reordered_joints

# Example usage
if __name__ == "__main__":
    input_file = "scripts/motion_converter/data/joint_angles_20250228_140351.txt"
    output_file = input_file.replace('.txt', '_resampled_reordered.txt')
    velocity_output_file = input_file.replace('.txt', '_joint_velocities.txt')
    
    resampled_data = resample_joint_poses(input_file, output_file, velocity_output_file)
    print(f"Successfully resampled data to 50Hz and reordered joints by type. Saved to {output_file}")
    
    # Print some statistics
    input_data = np.loadtxt(input_file)
    original_duration = input_data[-1, -1] - input_data[0, -1]
    original_freq = len(input_data) / original_duration
    print(f"Original frequency: {original_freq:.2f} Hz")
    print(f"Number of frames: {len(resampled_data)} frames")
    print(f"Duration: {original_duration:.2f} seconds")