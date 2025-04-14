import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

def load_endpoint_data(filename: str) -> np.ndarray:
    """
    Load endpoint positions from file.
    Result will be in the order: FL, FR, RL, RR (each with x, y, z coordinates)
    """
    print(f"Loading endpoint data from {filename}")
    data = np.loadtxt(filename)
    data = data[:, :12]  # Only take the first 12 columns
    
    # Reshape to intermediate format (frames, 12 coordinates)
    frames = data.shape[0]
    reshaped_data = data.reshape(frames, 12)
    
    # Initialize the new array with the reordered endpoints
    # Result will be (frames, 4 legs, 3 coordinates)
    result = np.zeros((frames, 4, 3))
    
    # Map the endpoints according to new order:
    # FL = 0, FR = 1, RL = 2, RR = 3
    # For each leg: [x, y, z]
    for i in range(4):
        # For each leg, copy x, y, z coordinates
        result[:, i, 0] = reshaped_data[:, i*3]     # x coordinate
        result[:, i, 1] = reshaped_data[:, i*3 + 1] # y coordinate
        result[:, i, 2] = reshaped_data[:, i*3 + 2] # z coordinate
    
    return result

def detect_stance_phase_using_dx(x_trajectory: np.ndarray) -> np.ndarray:
    """
    Detect stance phase (foot contact) based on x-direction movement.
    Stance is defined as when dx is negative (foot moving backward).
    Swing is defined as when dx is positive (foot moving forward).
    
    Args:
        x_trajectory: X-axis trajectory of a foot
    
    Returns:
        Boolean array where True indicates stance phase
    """
    # Calculate dx (change in x position)
    dx = np.zeros_like(x_trajectory)
    dx[1:] = x_trajectory[1:] - x_trajectory[:-1]
    
    # Set first element based on second element to avoid having no value
    # dx[0] = dx[1]
    dx[0] = x_trajectory[0] - x_trajectory[-1]
    
    # Detect stance when dx is negative (foot moving backward)
    stance = dx <= 0
    
    # Apply a small amount of smoothing to remove noise
    # This helps prevent rapid transitions that affect analysis
    smoothed_stance = np.copy(stance)
    
    # Simple smoothing: remove single-frame transitions
    for i in range(0, len(stance)-1):
        if stance[i] != stance[i-1] and stance[i] != stance[i+1]:
            smoothed_stance[i] = stance[i-1]
    
    # Print some statistics for debugging
    print(f"  Stance percentage: {np.mean(smoothed_stance)*100:.1f}%")
    
    return smoothed_stance

def calculate_swing_height(z_trajectory: np.ndarray, contact_mask: np.ndarray) -> float:
    """
    Calculate average swing height in meters.
    
    Args:
        z_trajectory: Z-axis trajectory of a foot
        contact_mask: Boolean array indicating stance phase
    
    Returns:
        Average swing height
    """
    # Get z values during swing phase
    swing_z = z_trajectory[~contact_mask]
    
    if len(swing_z) == 0:
        return 0.0
        
    # Find the minimum z value (stance height)
    min_z = np.min(z_trajectory)
    
    # Calculate maximum height during swing
    swing_height = np.max(swing_z) - min_z
    
    return swing_height

def analyze_gait(endpoint_data: np.ndarray, output_path: str, fps: float = 50.0) -> Tuple[Dict, Dict, Dict]:
    """
    Analyze quadruped gait from endpoint data.
    
    Args:
        endpoint_data: Endpoint positions with shape (frames, 4 legs, 3 coordinates)
        output_path: Path to save the contact patterns to
        fps: Frames per second
    
    Returns:
        Tuple containing:
        - contact_sequences: Dict mapping leg names to contact boolean arrays
        - swing_heights: Dict mapping leg names to swing heights
        - duty_factors: Dict mapping leg names to duty factors
    """
    leg_names = ["FL", "FR", "RL", "RR"]
    contact_sequences = {}
    swing_heights = {}
    duty_factors = {}
    
    # Process each leg
    for i, leg_name in enumerate(leg_names):
        print(f"\nAnalyzing {leg_name} leg:")
        
        # Extract x and z trajectories for this leg
        x_trajectory = endpoint_data[:, i, 0]
        z_trajectory = endpoint_data[:, i, 2]
        
        # Detect stance phase using x direction change
        contact = detect_stance_phase_using_dx(x_trajectory)
        contact_sequences[leg_name] = contact
        
        # Calculate duty factor (percentage of time in stance)
        duty_factor = np.mean(contact)
        duty_factors[leg_name] = duty_factor
        
        # Calculate swing height
        swing_height = calculate_swing_height(z_trajectory, contact)
        swing_heights[leg_name] = swing_height
        
        print(f"  Swing Height: {swing_height:.4f}m, Duty Factor: {duty_factor*100:.1f}%")
    
    # Count how many feet are in contact at each time step
    contact_array = np.array([contact_sequences[leg] for leg in leg_names])
    num_feet_in_contact = np.sum(contact_array, axis=0)
    
    # Analyze the distribution of feet in contact
    contact_distribution = [np.mean(num_feet_in_contact == i) for i in range(5)]
    
    # Print the distribution of feet in contact
    print("\nFeet in contact distribution:")
    for i in range(5):
        print(f"  {i} feet: {contact_distribution[i]*100:.1f}%")
    
    # Calculate average duty factor
    avg_duty_factor = np.mean([duty_factors[leg] for leg in leg_names])
    print(f"Average duty factor: {avg_duty_factor*100:.1f}%")
    
    # Print foot contact patterns for each time step
    print("\nFoot contact patterns [FL FR RL RR]:")
    # Create a text file in the same directory as the input file
    with open(output_path, 'w') as f:
        f.write("# Foot contact patterns [FL FR RL RR]\n")
        f.write("# 1 = contact (negative dx), 0 = no contact (positive dx)\n")
        for t in range(contact_array.shape[1]):
            # Convert boolean values to integers (1 for contact, 0 for no contact)
            pattern = contact_array[:, t].astype(int)
            line = f"[{pattern[0]} {pattern[1]} {pattern[2]} {pattern[3]}]"
            f.write(line + "\n")
            
            # Also print to console (limited to first 20 patterns to avoid flooding)
            if t < 20:
                print(f"[{line}]")
        print(f"... (saved {contact_array.shape[1]} patterns to {output_path})")
    
    return contact_sequences, swing_heights, duty_factors

def plot_gait_diagram(endpoint_data: np.ndarray, contact_sequences: Dict[str, np.ndarray], 
                     swing_heights: Dict[str, float], duty_factors: Dict[str, float], 
                     fps: float = 50.0):
    """
    Create a comprehensive visualization of the gait analysis.
    
    Args:
        endpoint_data: Endpoint positions with shape (frames, 4 legs, 3 coordinates)
        contact_sequences: Dict mapping leg names to contact boolean arrays
        swing_heights: Dict mapping leg names to swing heights
        duty_factors: Dict mapping leg names to duty factors
        fps: Frames per second
    """
    leg_names = ["FL", "FR", "RL", "RR"]
    colors = ['blue', 'red', 'green', 'orange']
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create 4 subplots: x trajectory, z trajectory, gait diagram, and summary statistics
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    
    # Time array
    time = np.arange(endpoint_data.shape[0]) / fps
    
    # Plot 1: X trajectory
    for i, (leg_name, color) in enumerate(zip(leg_names, colors)):
        x_trajectory = endpoint_data[:, i, 0]
        ax1.plot(time, x_trajectory, color=color, label=leg_name)
        
        # Highlight stance phases
        stance_mask = contact_sequences[leg_name]
        ax1.plot(time[stance_mask], x_trajectory[stance_mask], 'o', color=color, markersize=1)
    
    ax1.set_title('Foot X Trajectories')
    ax1.set_ylabel('X Position (m)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot dx (derivative of x) for one leg to better visualize the threshold
    leg_idx = 0  # FL leg
    x_trajectory = endpoint_data[:, leg_idx, 0]
    dx = np.zeros_like(x_trajectory)
    dx[1:] = x_trajectory[1:] - x_trajectory[:-1]
    dx[0] = x_trajectory[0] - x_trajectory[-1]  # Calculate using first and last frame
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, dx, 'k--', alpha=0.5, label='dx (FL)')
    ax1_twin.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax1_twin.set_ylabel('dx (m/frame)')
    ax1_twin.legend(loc='lower right')
    
    # Plot 2: Z trajectory
    for i, (leg_name, color) in enumerate(zip(leg_names, colors)):
        z_trajectory = endpoint_data[:, i, 2]
        ax2.plot(time, z_trajectory, color=color, label=leg_name)
        
        # Highlight stance phases
        stance_mask = contact_sequences[leg_name]
        ax2.plot(time[stance_mask], z_trajectory[stance_mask], 'o', color=color, markersize=1)
    
    ax2.set_title('Foot Z Trajectories')
    ax2.set_ylabel('Z Position (m)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Gait diagram (black = stance, white = swing)
    for i, leg_name in enumerate(leg_names):
        stance = contact_sequences[leg_name]
        # Create a colored bar for stance phases
        for j in range(len(stance)-1):
            if stance[j]:
                ax3.fill_between([time[j], time[j+1]], i+0.1, i+0.9, color='black')
    
    ax3.set_title('Gait Diagram (Black = Stance/Negative dx, White = Swing/Positive dx)')
    ax3.set_xlabel('Time (s)')
    ax3.set_yticks(np.arange(4) + 0.5)
    ax3.set_yticklabels(leg_names)
    ax3.grid(True)
    
    # Plot 4: Summary statistics as a table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Prepare table data
    table_data = []
    for i, leg_name in enumerate(leg_names):
        table_data.append([
            leg_name,
            f"{swing_heights[leg_name]:.4f} m",
            f"{duty_factors[leg_name]*100:.1f}%"
        ])
    
    # Create the table
    table = ax4.table(
        cellText=table_data,
        colLabels=["Leg", "Swing Height", "Duty Factor"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set overall title
    plt.suptitle('Quadruped Gait Analysis (Using dx for Stance Detection)', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, bottom=0.1, top=0.92)
    plt.show()

def main():
    # Default file path (change as needed)
    endpoint_file = 'scripts/motion_converter/data/joint_angles_20250327_180705_resampled_reordered_foot_endpoints.txt'
    
    # Generate output path for contact patterns
    import os
    output_dir = os.path.dirname(endpoint_file)
    output_name = os.path.splitext(os.path.basename(endpoint_file))[0] + '_contact_patterns_dx.txt'
    output_path = os.path.join(output_dir, output_name)
    
    # Load endpoint data
    endpoint_data = load_endpoint_data(endpoint_file)
    print(f"Loaded endpoint data with shape: {endpoint_data.shape}")
    
    # Set frames per second (adjust based on your data)
    fps = 50.0
    
    # Analyze gait
    contact_sequences, swing_heights, duty_factors = analyze_gait(endpoint_data, output_path, fps)
    
    # Print summary
    print(f"\nGait Analysis Summary:")
    
    print("\nSwing Heights:")
    for leg, height in swing_heights.items():
        print(f"  {leg}: {height:.4f} m")
    
    print("\nDuty Factors (% time in stance):")
    for leg, duty in duty_factors.items():
        print(f"  {leg}: {duty*100:.1f}%")
    
    # Create visualization
    plot_gait_diagram(endpoint_data, contact_sequences, swing_heights, duty_factors, fps)

if __name__ == "__main__":
    main()