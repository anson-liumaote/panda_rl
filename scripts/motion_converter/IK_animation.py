import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def load_joint_data(filename):
    """Load joint angles from file."""
    data = np.loadtxt(filename)
    data = data[:, :12]  # Only take the first 12 columns
    # Reshape to (frames, 4 legs, 3 joints)
    return data.reshape(-1, 4, 3)

def forward_kinematics(gamma, alpha, beta, leg_index, h=0.0365, hu=0.065, hl=0.065):
    """
    Calculate end-effector position given joint angles.
    gamma: hip joint angle (yaw)
    alpha: shoulder joint angle (pitch)
    beta: knee joint angle (pitch)
    leg_index: 0=FR, 1=FL, 2=HR, 3=HL
    """
    
    # Apply x-axis biases
    x_bias = 0.09067 if leg_index < 2 else -0.09067  # Front legs positive, hind legs negative
    
    # Apply y-axis biases
    y_bias = 0.085 if leg_index in [1, 3] else -0.085  # FL and HL positive, FR and HR negative
    
    h = h if leg_index in [1, 3] else -h  # FL and HL positive, FR and HR negative

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

class QuadrupedAnimation:
    def __init__(self, joint_data):
        self.joint_data = joint_data
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize lines for each leg with different colors
        self.leg_colors = ['red', 'blue', 'green', 'orange']  # FR, FL, HR, HL
        self.lines = [self.ax.plot([], [], [], f'{color}', label=f'Leg {i+1}')[0] 
                     for i, color in enumerate(self.leg_colors)]
        
        # Initialize base rectangle lines (black color)
        self.base_lines = [self.ax.plot([], [], [], 'k-')[0] for _ in range(4)]
        
        # Set axis limits
        self.ax.set_xlim([-0.2, 0.2])
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_zlim([-0.2, 0.2])
        
        # Labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadruped Leg Animation')
        
        # Add legend
        leg_names = ['Front Right', 'Front Left', 'Hind Right', 'Hind Left']
        self.ax.legend(self.lines, leg_names)

    def update(self, frame):
        # Get base positions for all legs
        base_positions = []
        for leg_idx in range(4):
            gamma, alpha, beta = self.joint_data[frame, leg_idx]
            p0, p1, p2, p3 = forward_kinematics(gamma, alpha, beta, leg_idx)
            base_positions.append(p0)
            
            # Update leg lines
            xs = [p0[0], p1[0], p2[0], p3[0]]
            ys = [p0[1], p1[1], p2[1], p3[1]]
            zs = [p0[2], p1[2], p2[2], p3[2]]
            self.lines[leg_idx].set_data(xs, ys)
            self.lines[leg_idx].set_3d_properties(zs)
        
        # Update base rectangle lines
        # Order: FR -> FL -> HL -> HR -> FR
        base_order = [0, 1, 3, 2, 0]  # Connect back to first point to close rectangle
        for i in range(4):
            start_idx = base_order[i]
            end_idx = base_order[i + 1]
            self.base_lines[i].set_data(
                [base_positions[start_idx][0], base_positions[end_idx][0]],
                [base_positions[start_idx][1], base_positions[end_idx][1]]
            )
            self.base_lines[i].set_3d_properties(
                [base_positions[start_idx][2], base_positions[end_idx][2]]
            )
        
        return self.lines + self.base_lines

    def animate(self):
        anim = FuncAnimation(
            self.fig, self.update,
            frames=len(self.joint_data),
            interval=12.5,
            blit=True
        )
        plt.show()

def main():
    # Load joint data
    joint_data = load_joint_data('scripts/motion_converter/joint_angles_20250220_171555.txt')
    
    # Create and run animation
    anim = QuadrupedAnimation(joint_data)
    anim.animate()

if __name__ == "__main__":
    main()