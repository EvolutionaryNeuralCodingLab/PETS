"""
Reproduction script for Figure 2b top section: Single-saccade examples.

This script loads the exported pickle data and recreates all saccade example plots
exactly as they appear in the paper.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the path
src_dir = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(src_dir))
from eye_tracking_system_tools.figures.plotting_functions import plot_angle_mapping, create_colorbar_pdf


def main():
    """Main function to reproduce all saccade example plots."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Load the pickle data
    pickle_file = script_dir / "saccade_examples.pickle"
    
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data for {data.get('figure_name', 'saccade_examples')}")
        print(f"Total examples: {data.get('total_examples', 0)}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get the list of saccade examples
    saccade_examples = data.get('saccade_examples', [])
    
    if not saccade_examples:
        print("Error: No saccade examples found in the pickle file")
        return
    
    # Group examples by time length (end_time - start_time) to create one colorbar per unique length
    from collections import defaultdict
    examples_by_length = defaultdict(list)
    
    for example in saccade_examples:
        start_time = example.get('start_time')
        end_time = example.get('end_time')
        time_length = end_time - start_time
        examples_by_length[time_length].append(example)
    
    print(f"Found {len(examples_by_length)} unique time lengths")
    
    # Create plots for each example (without colorbars)
    for i, example in enumerate(saccade_examples):
        label = example.get('label', f'example_{i+1}')
        start_time = example.get('start_time')
        end_time = example.get('end_time')
        figure_size = example.get('figure_size', (2.7, 1.7))
        xy_span = example.get('xy_span', 10)
        
        # Convert eye data dictionaries to DataFrames
        left_eye_dict = example.get('left_eye_data', {})
        right_eye_dict = example.get('right_eye_data', {})
        
        left_df = pd.DataFrame(left_eye_dict)
        right_df = pd.DataFrame(right_eye_dict)
        
        # Create export path
        export_path = script_dir / f"saccade_example_{label}_{start_time}_{end_time}.pdf"
        
        print(f"\nCreating plot for {label} (time: {start_time:.3f} - {end_time:.3f}s)")
        
        # Create the plot without colorbar
        plot_angle_mapping(
            start_time=start_time,
            end_time=end_time,
            left_df=left_df,
            right_df=right_df,
            figure_size=figure_size,
            export_path=str(export_path),
            xy_span=xy_span,
            create_colorbar=False  # Don't create individual colorbars
        )
    
    # Create one colorbar per unique time length
    print(f"\nCreating colorbars for unique time lengths...")
    for time_length, examples in examples_by_length.items():
        # Use the first example's parameters for the colorbar
        first_example = examples[0]
        start_time = first_example.get('start_time')
        end_time = first_example.get('end_time')
        figure_size = first_example.get('figure_size', (2.7, 1.7))
        
        # Create colorbar path
        colorbar_path = script_dir / f"colorbar_time_length_{time_length:.4f}s.pdf"
        
        create_colorbar_pdf(
            start_time=start_time,
            end_time=end_time,
            figure_height=figure_size[1],
            export_path=str(colorbar_path)
        )
    
    print(f"\nAll {len(saccade_examples)} saccade example plots created successfully!")
    print(f"Created {len(examples_by_length)} colorbars for unique time lengths")
    print(f"Plots saved to: {script_dir}")


if __name__ == "__main__":
    main()
