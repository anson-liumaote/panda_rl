#!/usr/bin/env python3
import numpy as np
import sys
from datetime import datetime

def transform_to_fr(x, y, z, leg_type):
    """
    Transform coordinates to front-right leg frame for IK calculation
    """
    if leg_type == 'fr':
        return x, y, z
    elif leg_type == 'fl':
        return x, -y, z
    elif leg_type == 'rr':
        return -x, y, z
    elif leg_type == 'rl':
        return -x, -y, z
    else:
        raise ValueError(f"Unknown leg type: {leg_type}")

def inverse_transform_angles(gamma, alpha, beta, leg_type):
    """
    Transform calculated angles back to each leg's frame
    """
    if leg_type == 'fr':
        return gamma, alpha, beta
    elif leg_type == 'fl':
        return -gamma, alpha, beta
    elif leg_type == 'rr':
        return gamma, -alpha, -beta
    elif leg_type == 'rl':
        return -gamma, -alpha, -beta
    else:
        raise ValueError(f"Unknown leg type: {leg_type}")

def inverse_kinematics(x, y, z):
    """
    Calculate joint angles from endpoint position
    """
    # h = 0.0375  # Hip offset
    h = 0.0375
    d = (y ** 2 + z ** 2) ** 0.5
    l = (d ** 2 - h ** 2) ** 0.5
    gamma1 = - np.arctan(h / l)
    gamma2 = - np.arctan(y / z)
    gamma = gamma2 - gamma1

    s = (l ** 2 + x ** 2) ** 0.5
    # Link lengths
    hu = 0.13
    hl = 0.13
    n = (s ** 2 - hl ** 2 - hu ** 2) / (2 * hu)
    beta = -np.arccos(n / hl)

    alpha1 = - np.arctan(x / l)
    alpha2 = np.arccos((hu + n) / s)
    alpha = alpha2 + alpha1

    return np.array([gamma, alpha, beta])

def normalize_foot_heights(endpoints):
    """
    Normalize the z-coordinates of the foot endpoints so that the lowest point
    is the same for all four legs across all frames.
    
    Args:
        endpoints (np.ndarray): Array of shape (num_frames, 12) containing
                               endpoints for all four legs (fl, fr, rl, rr)
    
    Returns:
        np.ndarray: Normalized endpoints with the same lowest z-coordinate
                    for all legs
    """
    num_frames = endpoints.shape[0]
    normalized_endpoints = endpoints.copy()
    
    # Find the minimum z-coordinate for each leg across all frames
    # Z-coordinates are at indices 2, 5, 8, 11 for fl, fr, rl, rr respectively
    z_indices = [2, 5, 8, 11]
    min_z_values = []
    
    for z_idx in z_indices:
        min_z_values.append(np.min(endpoints[:, z_idx]))
    
    # Find the overall minimum z-coordinate
    global_min_z = min(min_z_values)
    
    # Adjust each leg's z-coordinate to match the global minimum
    for i, z_idx in enumerate(z_indices):
        # Calculate the offset for this leg
        offset = min_z_values[i] - global_min_z
        
        # Apply the offset to all frames for this leg
        normalized_endpoints[:, z_idx] -= offset
    
    print(f"Normalized foot heights - Global minimum z: {global_min_z:.6f}")
    print(f"Original minimum z values: {min_z_values}")
    
    return normalized_endpoints

def process_endpoints(endpoint_file, output_pos_file, output_vel_file, normalize_heights=True):
    """
    Process endpoint positions to joint angles and velocities
    
    Args:
        endpoint_file (str): Path to the input endpoint file
        output_pos_file (str): Path to the output joint positions file
        output_vel_file (str): Path to the output joint velocities file
        normalize_heights (bool): Whether to normalize foot heights
    """
    # Load endpoint data from file
    endpoints = np.loadtxt(endpoint_file)
    
    # Normalize foot heights if requested
    if normalize_heights:
        print("Normalizing foot heights...")
        endpoints = normalize_foot_heights(endpoints)
        
        # Save normalized endpoints to a new file
        normalized_endpoints_file = endpoint_file.replace('.txt', '_normalized.txt')
        np.savetxt(
            normalized_endpoints_file,
            endpoints,
            fmt='%.6f',
            header='Normalized foot endpoints\n'
                   'Order: fl(xyz) fr(xyz) rl(xyz) rr(xyz)',
            comments=''
        )
        print(f"Normalized endpoints saved to: {normalized_endpoints_file}")
        
        # Also save a properly formatted version for direct use
        formatted_endpoints_file = endpoint_file.replace('.txt', '_normalized_formatted.txt')
        with open(formatted_endpoints_file, 'w') as f:
            for frame in endpoints:
                line = ' '.join([f"{coord:.6f}" for coord in frame])
                f.write(line + '\n')
        print(f"Formatted normalized endpoints saved to: {formatted_endpoints_file}")
    
    # Define leg origins (from original code)
    leg_origins = {
        'fr': (0.128, -0.055, 0),
        'fl': (0.128, 0.055, 0),
        'rr': (-0.128, -0.055, 0),
        'rl': (-0.128, 0.055, 0)
    }
    
    # Initialize arrays for joint positions and velocities
    num_frames = endpoints.shape[0]
    joint_positions = np.zeros((num_frames, 12))  # 12 joints
    
    # Process each frame
    for i in range(num_frames):
        frame_endpoints = endpoints[i]
        
        # Process each leg with the given order
        leg_order = [('fl', 0), ('fr', 3), ('rl', 6), ('rr', 9)]
        
        # Map to our code's expected leg types
        leg_type_map = {'fl': 'fl', 'fr': 'fr', 'rl': 'rl', 'rr': 'rr'}
        
        # Processed angles in our target order:
        # [fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf]
        processed_angles = np.zeros(12)
        
        for leg_name, start_idx in leg_order:
            # Extract coordinates
            x, y, z = frame_endpoints[start_idx:start_idx+3]
            
            # Get corresponding leg type in our code
            leg_type = leg_type_map[leg_name]
            
            # Remove leg origin bias before IK calculation
            origin_x, origin_y, origin_z = leg_origins[leg_type]
            x -= origin_x
            y -= origin_y
            z -= origin_z
            
            # Transform coordinates for IK calculation
            x_fr, y_fr, z_fr = transform_to_fr(x, y, z, leg_type)
            
            # Calculate angles
            gamma, alpha, beta = inverse_kinematics(x_fr, y_fr, z_fr)
            
            # Transform angles back to leg frame
            gamma, alpha, beta = inverse_transform_angles(gamma, alpha, beta, leg_type)
            
            # Map to output indices
            # [fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf]
            if leg_name == 'fl':
                processed_angles[0] = gamma    # fl_hip
                processed_angles[4] = alpha    # fl_thigh
                processed_angles[8] = beta     # fl_calf
            elif leg_name == 'fr':
                processed_angles[1] = gamma    # fr_hip
                processed_angles[5] = alpha    # fr_thigh
                processed_angles[9] = beta     # fr_calf
            elif leg_name == 'rl':
                processed_angles[2] = gamma    # rl_hip
                processed_angles[6] = alpha    # rl_thigh
                processed_angles[10] = beta    # rl_calf
            elif leg_name == 'rr':
                processed_angles[3] = gamma    # rr_hip
                processed_angles[7] = alpha    # rr_thigh
                processed_angles[11] = beta    # rr_calf
        
        # Store processed angles
        joint_positions[i] = processed_angles
    
    # Calculate joint velocities using finite differences
    dt = 0.02  # Assuming 50Hz sample rate (adjust if needed)
    joint_velocities = np.zeros_like(joint_positions)
    
    # For first frame, use last and second frames (cyclic boundary)
    joint_velocities[0] = (joint_positions[1] - joint_positions[-1]) / (2 * dt)
    
    # Central difference for middle frames
    for i in range(1, num_frames - 1):
        joint_velocities[i] = (joint_positions[i + 1] - joint_positions[i - 1]) / (2 * dt)
    
    # For last frame, use last-1 and first frames (cyclic boundary)
    joint_velocities[-1] = (joint_positions[0] - joint_positions[-2]) / (2 * dt)
    
    # Save joint positions
    np.savetxt(
        output_pos_file, 
        joint_positions,
        fmt='%.6f',
        header='Joint positions matrix\n'
               'Order: fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf'
    )
    
    # Save joint velocities
    np.savetxt(
        output_vel_file, 
        joint_velocities,
        fmt='%.6f',
        header='Joint velocities matrix\n'
               'Order: fl_hip, fr_hip, rl_hip, rr_hip, fl_thigh, fr_thigh, rl_thigh, rr_thigh, fl_calf, fr_calf, rl_calf, rr_calf'
    )
    
    print(f"Processed {num_frames} frames")
    print(f"Joint positions saved to: {output_pos_file}")
    print(f"Joint velocities saved to: {output_vel_file}")

def main():
    # Hardcoded file paths - modify these as needed
    endpoint_file = "scripts/motion_converter/data/joint_angles_20250228_140351_resampled_reordered_foot_endpoints.txt"  # Input file path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pos_file = endpoint_file.replace('.txt', '_joint_pos.txt')
    output_vel_file = endpoint_file.replace('.txt', '_joint_vel.txt')
    
    print(f"Processing endpoint file: {endpoint_file}")
    process_endpoints(endpoint_file, output_pos_file, output_vel_file, normalize_heights=True)

if __name__ == "__main__":
    main()