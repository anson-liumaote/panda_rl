import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_foot_animation(foot_targets, interval=100):
    """
    Creates a 3D animation of foot trajectories.
    
    Parameters
    ----------
    foot_targets : array-like
        Shape (num_frames, num_feet, 3) specifying the (x, y, z) coordinates of each foot
        for each frame.
    interval : int, optional
        Delay between frames in milliseconds (default=100).
    
    Returns
    -------
    ani : FuncAnimation
        Matplotlib animation object.
    """
    # Convert input to NumPy array
    data = np.array(foot_targets)
    num_frames = data.shape[0]
    num_feet = data.shape[1]
    
    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize lists for line and scatter objects
    lines = []
    current_scats = []
    
    # Prepare line and scatter for each foot
    for f in range(num_feet):
        # Empty line for the foot's path
        line, = ax.plot([], [], [], lw=2, color='gray')
        lines.append(line)
        
        # Scatter for the current position of the foot
        current_scatter = ax.scatter([], [], [], s=50, c='r')
        current_scats.append(current_scatter)

    # Set axis labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Determine axis limits from all data points
    all_points = data.reshape(-1, 3)
    x_min, x_max = np.min(all_points[:,0]), np.max(all_points[:,0])
    y_min, y_max = np.min(all_points[:,1]), np.max(all_points[:,1])
    z_min, z_max = np.min(all_points[:,2]), np.max(all_points[:,2])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Store each foot's trajectory in separate lists
    # foot_trajectories[f] = [x_list, y_list, z_list] for foot f
    foot_trajectories = [ [[], [], []] for _ in range(num_feet) ]

    def update_frame(frame):
        """Update function for each animation frame."""
        for f in range(num_feet):
            # Get current foot position
            x, y, z = data[frame, f]
            
            # Append current point to the foot's trajectory
            foot_trajectories[f][0].append(x)
            foot_trajectories[f][1].append(y)
            foot_trajectories[f][2].append(z)
            
            # Line: show all but the last point as the trailing path
            if len(foot_trajectories[f][0]) > 1:
                line_x = foot_trajectories[f][0][:-1]
                line_y = foot_trajectories[f][1][:-1]
                line_z = foot_trajectories[f][2][:-1]
            else:
                # If we have fewer than 2 points, no visible trailing line
                line_x, line_y, line_z = [], [], []
            
            # Update line data
            lines[f].set_data(line_x, line_y)
            lines[f].set_3d_properties(line_z)
            
            # Update current foot's scatter (the last point in its list)
            cur_x = foot_trajectories[f][0][-1]
            cur_y = foot_trajectories[f][1][-1]
            cur_z = foot_trajectories[f][2][-1]
            current_scats[f]._offsets3d = (np.array([cur_x]), 
                                           np.array([cur_y]), 
                                           np.array([cur_z]))

        ax.set_title(f"Frame {frame+1}/{num_frames}")
        return lines + current_scats

    # Create the animation
    ani = FuncAnimation(fig, update_frame, frames=range(num_frames), 
                        interval=interval, blit=True)

    return ani