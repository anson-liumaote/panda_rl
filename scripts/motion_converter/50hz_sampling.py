import numpy as np
from scipy.interpolate import interp1d

def resample_joint_poses(input_file, output_file, target_freq=50, blend_range=5):
    """
    Resample joint pose data to a target frequency using linear interpolation.
    Handles cyclic motion by blending start and end poses.
    
    Parameters:
    input_file (str): Path to input file containing joint poses with timestamps
    output_file (str): Path to save resampled joint poses
    target_freq (float): Target frequency in Hz (default: 50)
    blend_range (int): Number of frames to blend at the start/end (default: 5)
    """
    # Load the data
    data = np.loadtxt(input_file)
    
    # Extract timestamps from the last column
    timestamps = data[:, -1]
    joint_data = data[:, :-1]
    
    # Create cyclic data by repeating and blending
    # Add last few frames to start and first few frames to end
    joint_data_extended = np.vstack([
        joint_data[-blend_range:],  # Add last frames to start
        joint_data,
        joint_data[:blend_range]    # Add first frames to end
    ])
    
    # Create extended timestamps
    time_step = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    timestamps_extended = np.concatenate([
        timestamps[0] - np.arange(blend_range, 0, -1) * time_step,
        timestamps,
        timestamps[-1] + np.arange(1, blend_range + 1) * time_step
    ])
    
    # Calculate the duration and create new timestamps at target frequency
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_freq) + 1
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    
    # Create interpolation function for each joint using linear interpolation
    interpolators = [interp1d(timestamps_extended, joint_data_extended[:, i], kind='linear') 
                    for i in range(joint_data.shape[1])]
    
    print('aaa')
    # Generate resampled data
    resampled_joints = np.zeros((len(new_timestamps), joint_data.shape[1]))
    for i, interpolator in enumerate(interpolators):
        resampled_joints[:, i] = interpolator(new_timestamps)
    
    # Apply blending weights at the start and end
    # blend_frames = int(blend_range * (target_freq / len(timestamps)) * duration)
    # for i in range(blend_frames):
    #     weight = i / blend_frames
    #     resampled_joints[i] = (1 - weight) * resampled_joints[-blend_frames + i] + weight * resampled_joints[i]
    #     resampled_joints[-i-1] = weight * resampled_joints[blend_frames-i-1] + (1 - weight) * resampled_joints[-i-1]
    
    # Save the resampled data
    np.savetxt(output_file, resampled_joints, fmt='%.6f', delimiter=' ')
    
    return resampled_joints

# Example usage
if __name__ == "__main__":
   
    input_file = "scripts/motion_converter/joint_angles_20250228_140420.txt"
    output_file = input_file.replace('.txt', '_resampled.txt')
    
    resampled_data = resample_joint_poses(input_file, output_file)
    print(f"Successfully resampled data to 50Hz. Saved to {output_file}")
    
    # Print some statistics
    input_data = np.loadtxt(input_file)
    original_duration = input_data[-1, -1] - input_data[0, -1]
    original_freq = len(input_data) / original_duration
    print(f"Original frequency: {original_freq:.2f} Hz")
    print(f"Number of frames: {len(resampled_data)} frames")
    print(f"Duration: {original_duration:.2f} seconds")