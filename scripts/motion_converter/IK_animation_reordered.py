import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def load_joint_data(filename):
    """
    Load joint angles from file.
    New joint order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
    """
    data = np.loadtxt(filename)
    data = data[:, :12]  # Only take the first 12 columns
    
    # Reshape to intermediate format (frames, 12 joints)
    frames = data.shape[0]
    reshaped_data = data.reshape(frames, 12)
    
    # Initialize the new array with the reordered joints
    # Result will be (frames, 4 legs, 3 joints)
    result = np.zeros((frames, 4, 3))
    
    # Map the joints according to new order:
    # FL = 0, FR = 1, RL = 2, RR = 3
    # For each leg: [hip, thigh, calf]
    
    # Hip joints: 0-3
    result[:, 0, 0] = reshaped_data[:, 0]  # FL hip
    result[:, 1, 0] = reshaped_data[:, 1]  # FR hip
    result[:, 2, 0] = reshaped_data[:, 2]  # RL hip
    result[:, 3, 0] = reshaped_data[:, 3]  # RR hip
    
    # Thigh joints: 4-7
    result[:, 0, 1] = reshaped_data[:, 4]  # FL thigh
    result[:, 1, 1] = reshaped_data[:, 5]  # FR thigh
    result[:, 2, 1] = reshaped_data[:, 6]  # RL thigh
    result[:, 3, 1] = reshaped_data[:, 7]  # RR thigh
    
    # Calf joints: 8-11
    result[:, 0, 2] = reshaped_data[:, 8]   # FL calf
    result[:, 1, 2] = reshaped_data[:, 9]   # FR calf
    result[:, 2, 2] = reshaped_data[:, 10]  # RL calf
    result[:, 3, 2] = reshaped_data[:, 11]  # RR calf
    
    return result

def load_endpoint_data(filename):
    """
    Load endpoint positions from file.
    New endpoint order: FL, FR, RL, RR (each with x, y, z coordinates)
    """
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

def forward_kinematics(gamma, alpha, beta, leg_index, h=0.0375, hu=0.13, hl=0.13):
    """
    Calculate end-effector position given joint angles.
    gamma: hip joint angle (yaw)
    alpha: shoulder joint angle (pitch)
    beta: knee joint angle (pitch)
    leg_index: 0=FL, 1=FR, 2=RL, 3=RR
    """
    # Apply x-axis biases
    x_bias = 0.128 if leg_index < 2 else -0.128  # Front legs positive, hind legs negative
    
    # Apply y-axis biases (updated for new leg order)
    y_bias = 0.055 if leg_index in [0, 2] else -0.055  # FL and RL positive, FR and RR negative
    
    h = h if leg_index in [0, 2] else -h  # FL and RL positive, FR and RR negative

    # Base position with bias
    base_pos = np.array([x_bias, y_bias, 0])
    
    # Rotation matrices
    # R_gamma around x-axis
    R_gamma = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])
    
    # R_alpha around y-axis
    R_alpha = np.array([
        [np.cos(alpha), 0, np.sin(alpha)],
        [0, 1, 0],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])
    
    # R_beta around y-axis
    R_beta = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    
    # Hip position (vertical base)
    hip_vector = np.array([0, h, 0])
    p1 = base_pos + R_gamma @ hip_vector
    
    # Knee position (starting from vertical)
    upper_leg_vector = np.array([0, 0, -hu])  # Negative because it starts pointing down
    p2 = p1 + R_gamma @ R_alpha @ upper_leg_vector
    
    # Foot position
    lower_leg_vector = np.array([0, 0, -hl])  # Negative because it starts pointing down
    p3 = p2 + R_gamma @ R_alpha @ R_beta @ lower_leg_vector
    
    return base_pos, p1, p2, p3

class CombinedQuadrupedAnimation:
    def __init__(self, joint_data, endpoint_data):
        self.joint_data = joint_data
        self.endpoint_data = endpoint_data
        
        # Create a figure with two subplots side by side
        self.fig = plt.figure(figsize=(18, 8))
        
        # Left subplot for joint-based kinematics
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title('Forward Kinematics (Joint Angles)')
        
        # Right subplot for recorded endpoints
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title('Recorded Endpoints')
        
        # Initialize lines and markers for each leg with different colors (updated for new leg order)
        self.leg_colors = ['blue', 'red', 'green', 'orange']  # FL, FR, RL, RR
        self.leg_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
        
        # Joint-based lines (left plot)
        self.joint_lines = [self.ax1.plot([], [], [], f'{color}', label=f'{name}')[0] 
                           for color, name in zip(self.leg_colors, self.leg_names)]
        
        # Endpoint markers (right plot)
        self.endpoint_lines = [self.ax2.plot([], [], [], f'{color}', marker='o', label=f'{name}')[0] 
                              for color, name in zip(self.leg_colors, self.leg_names)]
        
        # Base rectangle lines for both plots (black color)
        self.base_lines1 = [self.ax1.plot([], [], [], 'k-')[0] for _ in range(4)]
        self.base_lines2 = [self.ax2.plot([], [], [], 'k-')[0] for _ in range(4)]
        
        # Additional lines for the endpoint plot to show paths
        self.path_lines = [self.ax2.plot([], [], [], f'{color}', linestyle='--', alpha=0.3)[0]
                          for color in self.leg_colors]
        
        # Paths storage for endpoint trails
        self.paths = [[] for _ in range(4)]
        self.path_length = 20  # Number of frames to show in the trail
        
        # Set axis limits for both plots
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([-0.3, 0.3])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        
        # Frame counter display
        self.frame_text = self.fig.text(0.5, 0.02, 'Frame: 0', ha='center')
        
        # Set figure title
        self.fig.suptitle('Quadruped Leg Animation - Comparison', fontsize=16)

    def update(self, frame):
        # Update frame counter
        self.frame_text.set_text(f'Frame: {frame}/{len(self.joint_data)-1}')
        
        # Get base positions for all legs
        joint_base_positions = []
        endpoint_base_positions = []
        
        # Draw the robot based on joint angles (left plot)
        for leg_idx in range(4):
            gamma, alpha, beta = self.joint_data[frame, leg_idx]
            p0, p1, p2, p3 = forward_kinematics(gamma, alpha, beta, leg_idx)
            joint_base_positions.append(p0)
            
            # Update leg lines for joint-based visualization
            xs = [p0[0], p1[0], p2[0], p3[0]]
            ys = [p0[1], p1[1], p2[1], p3[1]]
            zs = [p0[2], p1[2], p2[2], p3[2]]
            self.joint_lines[leg_idx].set_data(xs, ys)
            self.joint_lines[leg_idx].set_3d_properties(zs)
        
        # Draw the robot based on endpoint recordings (right plot)
        for leg_idx in range(4):
            # Get recorded endpoint
            endpoint = self.endpoint_data[frame, leg_idx]
            
            # Use the same base position as in the joint plot
            p0 = joint_base_positions[leg_idx]
            endpoint_base_positions.append(p0)
            
            # Add endpoint to path history
            self.paths[leg_idx].append(endpoint)
            if len(self.paths[leg_idx]) > self.path_length:
                self.paths[leg_idx].pop(0)
            
            # Draw line from base to endpoint
            self.endpoint_lines[leg_idx].set_data([p0[0], endpoint[0]], [p0[1], endpoint[1]])
            self.endpoint_lines[leg_idx].set_3d_properties([p0[2], endpoint[2]])
            
            # Update path trail
            if len(self.paths[leg_idx]) > 1:
                path_array = np.array(self.paths[leg_idx])
                self.path_lines[leg_idx].set_data(path_array[:, 0], path_array[:, 1])
                self.path_lines[leg_idx].set_3d_properties(path_array[:, 2])
        
        # Update base rectangle lines for joint-based plot
        # Order: FL -> FR -> RR -> RL -> FL (updated for new leg order)
        base_order = [0, 1, 3, 2, 0]  # Connect back to first point to close rectangle
        for i in range(4):
            start_idx = base_order[i]
            end_idx = base_order[i + 1]
            
            # For joint-based plot
            self.base_lines1[i].set_data(
                [joint_base_positions[start_idx][0], joint_base_positions[end_idx][0]],
                [joint_base_positions[start_idx][1], joint_base_positions[end_idx][1]]
            )
            self.base_lines1[i].set_3d_properties(
                [joint_base_positions[start_idx][2], joint_base_positions[end_idx][2]]
            )
            
            # For endpoint plot
            self.base_lines2[i].set_data(
                [endpoint_base_positions[start_idx][0], endpoint_base_positions[end_idx][0]],
                [endpoint_base_positions[start_idx][1], endpoint_base_positions[end_idx][1]]
            )
            self.base_lines2[i].set_3d_properties(
                [endpoint_base_positions[start_idx][2], endpoint_base_positions[end_idx][2]]
            )

        return (self.joint_lines + self.base_lines1 + 
                self.endpoint_lines + self.path_lines + self.base_lines2)

    def animate(self):
        anim = FuncAnimation(
            self.fig, self.update,
            frames=min(len(self.joint_data), len(self.endpoint_data)),
            interval=5,
            blit=True
        )
        plt.tight_layout()
        plt.show()
        
        return anim

def main():
    # Set the file paths directly
    joint_file = 'scripts/motion_converter/data/joint_angles_20250327_180705_resampled_reordered.txt'
    endpoint_file = 'scripts/motion_converter/data/joint_angles_20250327_180705_resampled_reordered_foot_endpoints.txt'  # Adjust this to match your endpoint file name
    
    # Load joint data and endpoint data
    print(f"Loading joint data from {joint_file}")
    joint_data = load_joint_data(joint_file)
    
    print(f"Loading endpoint data from {endpoint_file}")
    endpoint_data = load_endpoint_data(endpoint_file)
    
    print(f"Joint data shape: {joint_data.shape}, Endpoint data shape: {endpoint_data.shape}")
    
    # Create and run animation
    anim = CombinedQuadrupedAnimation(joint_data, endpoint_data)
    animation = anim.animate()
    
    # Uncomment the following to save the animation
    # animation.save('quadruped_animation.mp4', writer='ffmpeg', fps=20)

if __name__ == "__main__":
    main()