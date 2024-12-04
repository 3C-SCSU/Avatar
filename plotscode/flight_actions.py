import os
os.environ['R_HOME'] = "C:/Program Files/R/R-4.3.1"
import rpy2.robjects as robjects


def main():
    # Dictionary of actions and their file paths
    actions = {
        "left": "brainwaves-csv/left",
        "right": "brainwaves-csv/right",
        "backward": "brainwaves-csv/backward",
        "forward": "brainwaves-csv/forward",
        "land": "brainwaves-csv/land",
        "takeoff": "brainwaves-csv/takeoff",
    }
    # Path to R script
    r_script_path = "brainwave_analysis.R"
    try:
        # Load the R script
        print(f"Executing R script: {r_script_path}")
        robjects.r.source(r_script_path)
        print("R script executed successfully.")
    except Exception as e:
        print(f"An error occurred while executing the R script: {e}")
        return

    for action, file_path in actions.items():
        # Create an object of the BrainWaveAnalysis class
        robjects.r(f'BrainWaveAnalysis$new("{file_path}")')


if __name__ == "__main__":
    main()
