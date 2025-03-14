import numpy as np
from scipy.interpolate import interp1d

def resample_leg_endpoints(input_file, output_file, target_freq=50, blend_range=5):
    """
    Resample leg endpoint data to a target frequency using linear interpolation.
    Handles cyclic motion by blending start and end poses.
    Reorders legs from [fr[x,y,z] fl[x,y,z] hr[x,y,z] hl[x,y,z]]
    to [fl[x,y,z] fr[x,y,z] rl[x,y,z] rr[x,y,z]]
    
    Parameters:
    input_file (str): Path to input file containing leg endpoints with timestamps
    output_file (str): Path to save resampled leg endpoints
    target_freq (float): Target frequency in Hz (default: 50)
    blend_range (int): Number of frames to blend at the start/end (default: 5)
    """
    # Load the data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                data.append([float(val) for val in line.strip().split()])
    
    data = np.array(data)
    
    # Extract timestamps from the last column
    timestamps = data[:, -1]
    endpoint_data = data[:, :-1]
    
    # Create cyclic data by repeating and blending
    # Add last few frames to start and first few frames to end
    endpoints_extended = np.vstack([
        endpoint_data[-blend_range:],  # Add last frames to start
        endpoint_data,
        endpoint_data[:blend_range]    # Add first frames to end
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
    
    # Create interpolation function for each coordinate using linear interpolation
    interpolators = [interp1d(timestamps_extended, endpoints_extended[:, i], kind='linear') 
                    for i in range(endpoint_data.shape[1])]
    
    # Generate resampled data
    resampled_endpoints = np.zeros((len(new_timestamps), endpoint_data.shape[1]))
    for i, interpolator in enumerate(interpolators):
        resampled_endpoints[:, i] = interpolator(new_timestamps)
    
    # Reorder the leg endpoints 
    # Original order: fr[x,y,z] fl[x,y,z] hr[x,y,z] hl[x,y,z]
    # New order: fl[x,y,z] fr[x,y,z] rl[x,y,z] rr[x,y,z]
    
    reordered_endpoints = np.zeros_like(resampled_endpoints)
    
    # fl[x,y,z] (original indices 3,4,5) -> indices 0,1,2
    reordered_endpoints[:, 0:3] = resampled_endpoints[:, 3:6]
    
    # fr[x,y,z] (original indices 0,1,2) -> indices 3,4,5
    reordered_endpoints[:, 3:6] = resampled_endpoints[:, 0:3]
    
    # rl[x,y,z] (original hl[x,y,z]: indices 9,10,11) -> indices 6,7,8
    reordered_endpoints[:, 6:9] = resampled_endpoints[:, 9:12]
    
    # rr[x,y,z] (original hr[x,y,z]: indices 6,7,8) -> indices 9,10,11
    reordered_endpoints[:, 9:12] = resampled_endpoints[:, 6:9]
    
    # Save the resampled and reordered data
    np.savetxt(output_file, reordered_endpoints, fmt='%.6f', delimiter=' ')
    
    return reordered_endpoints

# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    input_file = "scripts/motion_converter/leg_endpoints_20250306_120153.txt"
    output_file = input_file.replace('.txt', '_resampled_reordered.txt')
    
    # Use the input file directly
    # Make sure the file exists and is readable before proceeding
    
    resampled_data = resample_leg_endpoints(input_file, output_file)
    
    # Print some statistics
    data = np.loadtxt(input_file, comments='#')
    original_duration = data[-1, -1] - data[0, -1]
    original_freq = len(data) / original_duration
    
    print(f"Successfully resampled leg endpoints to 50Hz and reordered legs. Saved to {output_file}")
    print(f"Original frequency: {original_freq:.2f} Hz")
    print(f"Number of frames: {len(resampled_data)} frames")
    print(f"Duration: {original_duration:.2f} seconds")
    
    # Show first few lines of the resampled and reordered data
    print("\nFirst 5 lines of resampled and reordered data:")
    for i in range(min(5, len(resampled_data))):
        print(" ".join([f"{val:.6f}" for val in resampled_data[i]]))