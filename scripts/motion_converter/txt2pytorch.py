import torch
import numpy as np

class JointAnglesReader:
    """
    A class to read joint angles data from a text file and convert to PyTorch tensors.
    The data format is expected to be space-separated values with 12 joint angles per line.
    When reaching the end of file, it automatically loops back to the beginning.
    Each pose is extended to 10 identical instances.
    """
    
    NUM_JOINTS = 12  # Constant number of joints per frame
    NUM_INSTANCES = 1  # Number of instances to repeat each pose
    
    def __init__(self, file_path):
        """
        Initialize the reader with the file path.
        
        Args:
            file_path (str): Path to the joint angles text file
        """
        self.file_path = file_path
        self.current_line = 0
        
        # Read all lines at initialization
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(self.lines)
        
    def read_next_pose(self):
        """
        Read the next pose from the file and return as a PyTorch tensor.
        When reaching the end of file, loops back to the beginning.
        Each pose is repeated 10 times.
        
        Returns:
            torch.Tensor: Tensor of shape (10, 12) containing joint angles
        """
        # If we reach the end, loop back to beginning
        if self.current_line >= self.total_frames:
            self.current_line = 0
            
        # Read the current line and split into values
        line = self.lines[self.current_line]
        values = line.strip().split()
        
        # Convert to float and create tensor
        joint_angles = [float(x) for x in values]
        single_pose_tensor = torch.tensor(joint_angles).reshape(1, self.NUM_JOINTS)
        
        # Repeat the pose 10 times
        pose_tensor = single_pose_tensor.repeat(self.NUM_INSTANCES, 1)
        
        # Increment line counter
        self.current_line += 1
        
        return pose_tensor
    
    def reset(self):
        """Reset the reader to the beginning of the file."""
        self.current_line = 0
    
    def get_current_frame(self):
        """Return the current frame number."""
        return self.current_line
    
    def get_total_frames(self):
        """Return the total number of frames in the file."""
        return self.total_frames

# Example usage:
reader = JointAnglesReader('triceratops_robot/triceratops_base/joint_angles_20250219_150134.txt')
for i in range(400):
    poses = reader.read_next_pose()  # Returns tensor of shape (10, 12)
    print(poses)
# reader.reset()  # Go back to beginning