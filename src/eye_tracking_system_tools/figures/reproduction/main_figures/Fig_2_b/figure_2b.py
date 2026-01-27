"""
Reproduction script for Figure 2b: Simultaneous eye angle tracking for both eyes.

This script loads the exported pickle data and recreates the figure exactly as it appears in the paper.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the path
src_dir = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(src_dir))
from eye_tracking_system_tools.figures.plotting_functions import plot_zoomed_in_with_head_rate


def main():
    """Main function to reproduce Figure 2b."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Load the pickle data
    pickle_file = script_dir / "figure_2b.pickle"
    output_path = script_dir / "figure_2b.pdf"
    
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data for {data.get('figure_name', 'figure_2b')}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract data
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    traces = data.get('traces', ['center_x', 'center_y'])
    figure_params = data.get('figure_params', {})
    
    # Convert eye data dictionaries to DataFrames
    left_eye_dict = data.get('left_eye_data', {})
    right_eye_dict = data.get('right_eye_data', {})
    
    left_df = pd.DataFrame(left_eye_dict)
    right_df = pd.DataFrame(right_eye_dict)
    
    # Extract event times (convert to arrays if they exist)
    left_ms = data.get('left_saccade_ms')
    right_ms = data.get('right_saccade_ms')
    head_movements_ms = data.get('head_movements_ms')
    behavior_state = data.get('behavior_state')
    
    # Convert behavior_state to DataFrame if it's a list
    if behavior_state is not None:
        if isinstance(behavior_state, list):
            behavior_state_df = pd.DataFrame(behavior_state)
        elif isinstance(behavior_state, pd.DataFrame):
            behavior_state_df = behavior_state
        else:
            behavior_state_df = None
    else:
        behavior_state_df = None
    
    # Convert to numpy arrays if they are lists
    if left_ms is not None and not isinstance(left_ms, np.ndarray):
        left_ms = np.array(left_ms) if len(left_ms) > 0 else None
    if right_ms is not None and not isinstance(right_ms, np.ndarray):
        right_ms = np.array(right_ms) if len(right_ms) > 0 else None
    if head_movements_ms is not None and not isinstance(head_movements_ms, np.ndarray):
        head_movements_ms = np.array(head_movements_ms) if len(head_movements_ms) > 0 else None
    
    # Call the plotting function with parameters from the pickle file
    plot_zoomed_in_with_head_rate(
        start_time=start_time,
        end_time=end_time,
        traces=traces,
        left_df=left_df,
        right_df=right_df,
        left_ms=left_ms,
        right_ms=right_ms,
        head_movements_ms=head_movements_ms,
        behavior_state_df=behavior_state_df,
        export_path=str(output_path),
        **figure_params
    )
    
    print(f"Figure saved to {output_path}")
    print("Figure 2b reproduction complete!")


if __name__ == "__main__":
    main()
