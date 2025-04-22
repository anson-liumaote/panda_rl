import numpy as np
from scipy.interpolate import interp1d

def resample_joint_poses(input_file, output_file, velocity_output_file=None, source_freq=100, target_freq=50):
    """
    Resample joint pose data from source frequency to target frequency using linear interpolation.
    Reorders joints from [fl(γαβ), fr(γαβ), rr(γαβ), rl(γαβ)] to
    [fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf]
    
    Parameters:
    input_file (str): Path to input file containing joint poses with timestamps
    output_file (str): Path to save resampled joint poses
    velocity_output_file (str, optional): Path to save joint velocities
    source_freq (float): Source frequency in Hz (default: 100)
    target_freq (float): Target frequency in Hz (default: 50)
    """
    # Load the data
    data = np.loadtxt(input_file)
    
    # Check if the last column is timestamps
    has_timestamps = False
    if data.shape[1] > 12:  # Assuming joint data has at least 12 columns
        # Extract timestamps from the last column
        timestamps = data[:, -1]
        joint_data = data[:, :-1]
        has_timestamps = True
    else:
        # Create timestamps based on the assumed source frequency
        joint_data = data
        timestamps = np.linspace(0, (len(joint_data) - 1) / source_freq, len(joint_data))
    
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
    # Input order: fl[gamma,alpha,beta] fr[gamma,alpha,beta] rr[gamma,alpha,beta] rl[gamma,alpha,beta]
    # New order: fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf
    
    reordered_joints = np.zeros_like(resampled_joints)
    
    # Mapping from input to new indices:
    # fl_hip (input fl_gamma: index 0) -> index 0
    reordered_joints[:, 0] = resampled_joints[:, 0]
    
    # fr_hip (input fr_gamma: index 3) -> index 1
    reordered_joints[:, 1] = resampled_joints[:, 3]
    
    # rl_hip (input rl_gamma: index 9) -> index 2
    reordered_joints[:, 2] = resampled_joints[:, 6]
    
    # rr_hip (input rr_gamma: index 6) -> index 3
    reordered_joints[:, 3] = resampled_joints[:, 9]
    
    # fl_thigh (input fl_alpha: index 1) -> index 4
    reordered_joints[:, 4] = resampled_joints[:, 1]
    
    # fr_thigh (input fr_alpha: index 4) -> index 5
    reordered_joints[:, 5] = resampled_joints[:, 4]
    
    # rl_thigh (input rl_alpha: index 10) -> index 6
    reordered_joints[:, 6] = resampled_joints[:, 7]
    
    # rr_thigh (input rr_alpha: index 7) -> index 7
    reordered_joints[:, 7] = resampled_joints[:, 10]
    
    # fl_calf (input fl_beta: index 2) -> index 8
    reordered_joints[:, 8] = resampled_joints[:, 2]
    
    # fr_calf (input fr_beta: index 5) -> index 9
    reordered_joints[:, 9] = resampled_joints[:, 5]
    
    # rl_calf (input rl_beta: index 11) -> index 10
    reordered_joints[:, 10] = resampled_joints[:, 8]
    
    # rr_calf (input rr_beta: index 8) -> index 11
    reordered_joints[:, 11] = resampled_joints[:, 11]
    
    # Save the resampled and reordered data
    np.savetxt(output_file, reordered_joints, fmt='%.6f', delimiter=' ')
    print(f"Resampled and reordered data saved to {output_file}")
    
    # Calculate joint velocities if requested
    if velocity_output_file:
        # Calculate time difference between frames (should be constant for resampled data)
        dt = 1.0 / target_freq
        
        # Calculate velocities (rad/s) using central difference
        velocities = np.zeros_like(reordered_joints)
        
        # Central difference for middle frames
        velocities[1:-1] = (reordered_joints[2:] - reordered_joints[:-2]) / (2 * dt)
        
        # Forward difference for first frame
        velocities[0] = (reordered_joints[1] - reordered_joints[0]) / dt
        
        # Central difference for last frames
        # using loop_start frame - last second frame
        velocities[-1] = (reordered_joints[113] - reordered_joints[-2]) / (2 * dt)
        
        # Save velocities to file
        np.savetxt(velocity_output_file, velocities, fmt='%.6f', delimiter=' ')
        print(f"Joint velocities saved to {velocity_output_file}")
    
    return reordered_joints

# Example usage
if __name__ == "__main__":
    input_file = "scripts/motion_converter/data/optimized_joint_positions.txt"
    output_file = input_file.replace('.txt', '_resampled_reordered.txt')
    velocity_output_file = input_file.replace('.txt', '_joint_velocities.txt')
    
    resampled_data = resample_joint_poses(input_file, output_file, velocity_output_file)
    print(f"Successfully resampled data from 100Hz to 50Hz and reordered joints. Saved to {output_file}")
    
    # Print some statistics
    input_data = np.loadtxt(input_file)
    if input_data.shape[1] > 12:
        original_duration = input_data[-1, -1] - input_data[0, -1]
    else:
        original_duration = (len(input_data) - 1) / 100  # Assuming 100Hz source
    
    print(f"Number of frames: {len(resampled_data)} frames")
    print(f"Duration: {original_duration:.2f} seconds")