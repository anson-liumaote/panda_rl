import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('scripts/motion_converter/joint_angles_20250228_140351_resampled_reordered_foot_endpoints.txt')

# Extract the z-coordinates for each foot
# According to the description: fl(xyz) fr(xyz) rl(xyz) rr(xyz)
fl_z = data[:, 2]  # Front Left Z (3rd column)
fr_z = data[:, 5]  # Front Right Z (6th column)
rl_z = data[:, 8]  # Rear Left Z (9th column)
rr_z = data[:, 11] # Rear Right Z (12th column)

# Function to compute threshold for contact detection
def compute_threshold(z_values, percentile=0.3):
    """
    Compute threshold for foot contact detection.
    We use a small percentile above the minimum value.
    """
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    z_range = z_max - z_min
    return z_min + z_range * percentile

# Calculate thresholds for each foot
thresholds = {
    'fl': compute_threshold(fl_z),
    'fr': compute_threshold(fr_z),
    'rl': compute_threshold(rl_z),
    'rr': compute_threshold(rr_z)
}

# Detect contact phases
def detect_contacts(z_values, threshold):
    """
    Detect contact phases by comparing z values to threshold.
    Returns lists of contact and air phases.
    """
    contact = z_values <= threshold
    contact_phases = []
    air_phases = []
    
    # Find contiguous contact phases
    in_contact = False
    start_idx = 0
    
    for i, is_contact in enumerate(contact):
        # State transition: air to contact
        if not in_contact and is_contact:
            start_idx = i
            in_contact = True
        # State transition: contact to air
        elif in_contact and not is_contact:
            contact_phases.append({'start': start_idx, 'end': i-1})
            in_contact = False
    
    # If still in contact at the end
    if in_contact:
        contact_phases.append({'start': start_idx, 'end': len(z_values)-1})
    
    # Find air phases (inverse of contact phases)
    if not contact_phases:
        air_phases = [{'start': 0, 'end': len(z_values)-1}]
    else:
        # Add initial air phase if needed
        if contact_phases[0]['start'] > 0:
            air_phases.append({'start': 0, 'end': contact_phases[0]['start']-1})
        
        # Add intermediate air phases
        for i in range(len(contact_phases)-1):
            air_phases.append({
                'start': contact_phases[i]['end']+1,
                'end': contact_phases[i+1]['start']-1
            })
        
        # Add final air phase if needed
        if contact_phases[-1]['end'] < len(z_values)-1:
            air_phases.append({'start': contact_phases[-1]['end']+1, 'end': len(z_values)-1})
    
    return {'contact': contact_phases, 'air': air_phases}

# Detect phases for all feet
phases = {
    'fl': detect_contacts(fl_z, thresholds['fl']),
    'fr': detect_contacts(fr_z, thresholds['fr']),
    'rl': detect_contacts(rl_z, thresholds['rl']),
    'rr': detect_contacts(rr_z, thresholds['rr'])
}

# Calculate statistics
def calculate_stats(phases, total_frames):
    """
    Calculate contact and air time statistics.
    """
    contact_frames = sum(phase['end'] - phase['start'] + 1 for phase in phases['contact'])
    air_frames = sum(phase['end'] - phase['start'] + 1 for phase in phases['air'])
    
    stats = {
        'contact_time_pct': (contact_frames / total_frames) * 100,
        'air_time_pct': (air_frames / total_frames) * 100,
        'avg_contact_duration': contact_frames / len(phases['contact']) if phases['contact'] else 0,
        'avg_air_duration': air_frames / len(phases['air']) if phases['air'] else 0
    }
    
    return stats

total_frames = len(data)
stats = {
    'fl': calculate_stats(phases['fl'], total_frames),
    'fr': calculate_stats(phases['fr'], total_frames),
    'rl': calculate_stats(phases['rl'], total_frames),
    'rr': calculate_stats(phases['rr'], total_frames)
}

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})

# Timeline plot (top subplot)
ax1.set_title('Foot Contact Timeline', fontsize=16)

# Helper function to visualize phases
def visualize_phases(ax, y_pos, phases, color, label):
    for phase in phases:
        ax.fill_between(
            range(phase['start'], phase['end'] + 1), 
            [y_pos] * (phase['end'] - phase['start'] + 1), 
            [y_pos + 0.8] * (phase['end'] - phase['start'] + 1), 
            color=color, alpha=0.7, label=label if phase == phases[0] else ""
        )

# Plot contact phases for each foot in the order FL, RL, FR, RR
visualize_phases(ax1, 0, phases['fl']['contact'], 'blue', 'Front Left')
visualize_phases(ax1, 1, phases['rl']['contact'], 'green', 'Rear Left')
visualize_phases(ax1, 2, phases['fr']['contact'], 'red', 'Front Right')
visualize_phases(ax1, 3, phases['rr']['contact'], 'purple', 'Rear Right')

ax1.set_yticks([0.4, 1.4, 2.4, 3.4])
ax1.set_yticklabels(['FL', 'RL', 'FR', 'RR'], fontsize=12)
ax1.set_xlabel('Frame', fontsize=12)
ax1.set_xlim(0, total_frames)
ax1.set_ylim(0, 4)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Bar chart (bottom subplot)
legs = ['Front Left', 'Rear Left', 'Front Right', 'Rear Right']
contact_percentages = [
    stats['fl']['contact_time_pct'],
    stats['rl']['contact_time_pct'],
    stats['fr']['contact_time_pct'],
    stats['rr']['contact_time_pct']
]
air_percentages = [
    stats['fl']['air_time_pct'],
    stats['rl']['air_time_pct'],
    stats['fr']['air_time_pct'],
    stats['rr']['air_time_pct']
]

x = np.arange(len(legs))
width = 0.35

ax2.set_title('Contact Time vs. Air Time Percentage', fontsize=16)
bars1 = ax2.bar(x - width/2, contact_percentages, width, label='Contact Time %', color='blue')
bars2 = ax2.bar(x + width/2, air_percentages, width, label='Air Time %', color='orange')

ax2.set_xticks(x)
ax2.set_xticklabels(legs, fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

# Add text labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

# Print detailed statistics
print("Foot Contact Analysis Statistics:")
for leg, leg_name in zip(['fl', 'rl', 'fr', 'rr'], legs):
    print(f"\n{leg_name}:")
    print(f"  Contact time: {stats[leg]['contact_time_pct']:.1f}%")
    print(f"  Air time: {stats[leg]['air_time_pct']:.1f}%")
    print(f"  Average contact duration: {stats[leg]['avg_contact_duration']:.1f} frames")
    print(f"  Average air duration: {stats[leg]['avg_air_duration']:.1f} frames")
    print(f"  Number of contact phases: {len(phases[leg]['contact'])}")
    print(f"  Number of air phases: {len(phases[leg]['air'])}")

# Add an overall title
fig.suptitle('Foot Contact and Air Time Analysis', fontsize=18, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for overall title
plt.savefig('foot_contact_analysis.png', dpi=300)
plt.show()