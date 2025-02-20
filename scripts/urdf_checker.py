import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

class URDFChecker:
    def __init__(self, parameter_pairs: List[Tuple[str, ...]]):
        self.parameter_pairs = parameter_pairs
        self.parameters = {}

    def parse_urdf(self, urdf_file: str) -> None:
        """Parse the URDF file and extract COM and inertia parameters."""
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        for link in root.findall(".//link"):
            link_name = link.get('name')
            inertial = link.find('inertial')
            
            if inertial is not None:
                # Get COM
                origin = inertial.find('origin')
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0').split()
                    self.parameters[f"{link_name}_com"] = float(xyz[2])  # Only z-axis
                
                # Get principal inertia
                inertia = inertial.find('inertia')
                if inertia is not None:
                    self.parameters[f"{link_name}_inertia"] = {
                        'ixx': float(inertia.get('ixx', 0)),
                        'iyy': float(inertia.get('iyy', 0)),
                        'izz': float(inertia.get('izz', 0))
                    }

    def check_com_z_signs(self) -> Dict[str, bool]:
        """Check if COM z-coordinates in each pair have the same sign."""
        results = {}
        
        for pair in self.parameter_pairs:
            values = []
            for param in pair:
                com_key = f"{param}_com"
                if com_key in self.parameters:
                    values.append(self.parameters[com_key])
            
            if values:
                all_positive = all(v >= 0 for v in values)
                all_negative = all(v <= 0 for v in values)
                same_sign = all_positive or all_negative
                
                results[f"{'-'.join(pair)}_z"] = same_sign
        
        return results

    def check_principal_inertia_scale(self, max_ratio: float = 10.0) -> Dict[str, bool]:
        """Check if principal inertia values (xx, yy, zz) are within scale ratio."""
        results = {}
        
        for pair in self.parameter_pairs:
            for component in ['ixx', 'iyy', 'izz']:
                values = []
                for param in pair:
                    inertia_key = f"{param}_inertia"
                    if inertia_key in self.parameters:
                        values.append(self.parameters[inertia_key][component])
                
                if values:
                    min_val = min(abs(v) for v in values if v != 0)
                    max_val = max(abs(v) for v in values if v != 0)
                    
                    ratio = max_val / min_val if min_val > 0 else float('inf')
                    within_scale = ratio <= max_ratio
                    
                    results[f"{'-'.join(pair)}_{component}"] = within_scale
        
        return results

def get_auto_pairs(urdf_file: str) -> List[Tuple[str, ...]]:
    """Automatically detect parameter pairs from URDF file."""
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    # Common patterns for quadruped robots
    patterns = {
        'base': ['fr_Link', 'fl_Link', 'br_Link', 'bl_Link'],
        'upper': ['fru_Link', 'flu_Link', 'bru_Link', 'blu_Link'],
        'lower': ['frd_Link', 'fld_Link', 'brd_Link', 'bld_Link'],
        'foot': ['frf_Link', 'flf_Link', 'brf_Link', 'blf_Link']
    }
    
    pairs = []
    for pattern_group in patterns.values():
        # Check if all links in the pattern exist in the URDF
        if all(root.find(f".//link[@name='{link}']") is not None for link in pattern_group):
            pairs.append(tuple(pattern_group))
    
    return pairs

def main():
    """Main function to run the URDF checker."""
    print("=== URDF COM and Inertia Checker ===")
    
    # Get URDF file path
    urdf_file = input("\nEnter the path to your URDF file: ").strip()
    
    # Get parameter pairs
    print("\nEnter 'auto' for automatic parameter detection or press Enter to input manually:")
    choice = input().strip().lower()
    
    pairs = []
    if choice == 'auto':
        try:
            pairs = get_auto_pairs(urdf_file)
            print("\nDetected parameter pairs:")
            for pair in pairs:
                print("  " + " ".join(pair))
        except Exception as e:
            print(f"Error detecting pairs: {str(e)}")
            return
    else:
        print("\nEnter parameter pairs (one group per line, space-separated).")
        print("Example: fr_Link fl_Link br_Link bl_Link")
        print("Enter an empty line when done.")
        
        while True:
            line = input("\nEnter parameters (or press Enter to finish): ").strip()
            if not line:
                break
            pairs.append(tuple(line.split()))
    
    if not pairs:
        print("No parameter pairs found or entered. Exiting.")
        return
    
    try:
        checker = URDFChecker(pairs)
        checker.parse_urdf(urdf_file)
        
        # Check COM z-axis signs
        com_results = checker.check_com_z_signs()
        print("\nCOM Z-Axis Sign Check Results:")
        for param, passed in com_results.items():
            print(f"  {param}: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        # Check principal inertia scale
        inertia_results = checker.check_principal_inertia_scale()
        print("\nPrincipal Inertia Scale Check Results:")
        for param, passed in inertia_results.items():
            print(f"  {param}: {'✓ PASSED' if passed else '✗ FAILED'}")
            
    except ET.ParseError:
        print(f"Error: Could not parse URDF file '{urdf_file}'")
    except FileNotFoundError:
        print(f"Error: URDF file '{urdf_file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()