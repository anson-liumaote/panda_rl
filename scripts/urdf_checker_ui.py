import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import os

class URDFChecker:
    # [Previous URDFChecker class implementation remains the same]
    def __init__(self, parameter_pairs: List[Tuple[str, ...]]):
        self.parameter_pairs = parameter_pairs
        self.parameters = {}

    def parse_urdf(self, urdf_file: str) -> None:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        for link in root.findall(".//link"):
            link_name = link.get('name')
            inertial = link.find('inertial')
            
            if inertial is not None:
                origin = inertial.find('origin')
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0').split()
                    self.parameters[f"{link_name}_com"] = float(xyz[2])
                
                inertia = inertial.find('inertia')
                if inertia is not None:
                    self.parameters[f"{link_name}_inertia"] = {
                        'ixx': float(inertia.get('ixx', 0)),
                        'iyy': float(inertia.get('iyy', 0)),
                        'izz': float(inertia.get('izz', 0))
                    }

    def check_com_z_signs(self) -> Dict[str, bool]:
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

class URDFCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("URDF Parameter Checker")
        self.root.geometry("500x600")
        
        # Variables
        self.urdf_file = tk.StringVar()
        self.pairs_text = tk.StringVar()
        self.auto_detect = tk.BooleanVar(value=True)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="URDF File", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Entry(file_frame, textvariable=self.urdf_file, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1, padx=5)
        
        # Parameter pairs
        pairs_frame = ttk.LabelFrame(main_frame, text="Parameter Pairs", padding="5")
        pairs_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(pairs_frame, text="Auto Detect", variable=self.auto_detect, 
                       value=True, command=self.toggle_pairs_input).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(pairs_frame, text="Manual Input", variable=self.auto_detect, 
                       value=False, command=self.toggle_pairs_input).grid(row=0, column=1, padx=5)
        
        self.pairs_text_widget = scrolledtext.ScrolledText(pairs_frame, height=6, width=60)
        self.pairs_text_widget.grid(row=1, column=0, columnspan=2, pady=5)
        self.pairs_text_widget.insert(tk.END, "fr_Link fl_Link br_Link bl_Link\nfru_Link flu_Link bru_Link blu_Link\nfrd_Link fld_Link brd_Link bld_Link\nfrf_Link flf_Link brf_Link blf_Link")
        
        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Check Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=60)
        self.results_text.grid(row=0, column=0, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Run Check", command=self.run_check).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_results).grid(row=0, column=1, padx=5)
        
        self.toggle_pairs_input()
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("URDF files", "*.urdf"), ("XML files", "*.xml"), ("All files", "*.*")]
        )
        if filename:
            self.urdf_file.set(filename)
            
    def toggle_pairs_input(self):
        if self.auto_detect.get():
            self.pairs_text_widget.config(state='disabled')
        else:
            self.pairs_text_widget.config(state='normal')
            
    def get_parameter_pairs(self):
        if self.auto_detect.get():
            try:
                tree = ET.parse(self.urdf_file.get())
                root = tree.getroot()
                patterns = {
                    'base': ['fr_Link', 'fl_Link', 'br_Link', 'bl_Link'],
                    'upper': ['fru_Link', 'flu_Link', 'bru_Link', 'blu_Link'],
                    'lower': ['frd_Link', 'fld_Link', 'brd_Link', 'bld_Link'],
                    'foot': ['frf_Link', 'flf_Link', 'brf_Link', 'blf_Link']
                }
                pairs = []
                for pattern_group in patterns.values():
                    if all(root.find(f".//link[@name='{link}']") is not None for link in pattern_group):
                        pairs.append(tuple(pattern_group))
                return pairs
            except Exception as e:
                self.show_error(f"Error detecting pairs: {str(e)}")
                return []
        else:
            pairs = []
            for line in self.pairs_text_widget.get('1.0', tk.END).strip().split('\n'):
                if line.strip():
                    pairs.append(tuple(line.strip().split()))
            return pairs
            
    def run_check(self):
        if not self.urdf_file.get():
            self.show_error("Please select a URDF file first.")
            return
            
        pairs = self.get_parameter_pairs()
        if not pairs:
            self.show_error("No parameter pairs found or entered.")
            return
            
        try:
            checker = URDFChecker(pairs)
            checker.parse_urdf(self.urdf_file.get())
            
            # Clear previous results
            self.results_text.delete('1.0', tk.END)
            
            # Check COM z-axis signs
            self.results_text.insert(tk.END, "COM Z-Axis Sign Check Results:\n")
            com_results = checker.check_com_z_signs()
            for param, passed in com_results.items():
                result = "✓ PASSED" if passed else "✗ FAILED"
                self.results_text.insert(tk.END, f"  {param}: {result}\n")
            
            self.results_text.insert(tk.END, "\nPrincipal Inertia Scale Check Results:\n")
            inertia_results = checker.check_principal_inertia_scale()
            for param, passed in inertia_results.items():
                result = "✓ PASSED" if passed else "✗ FAILED"
                self.results_text.insert(tk.END, f"  {param}: {result}\n")
                
        except Exception as e:
            self.show_error(f"Error running checks: {str(e)}")
            
    def clear_results(self):
        self.results_text.delete('1.0', tk.END)
        
    def show_error(self, message):
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, f"Error: {message}\n")

def main():
    root = tk.Tk()
    app = URDFCheckerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()