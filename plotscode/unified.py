import os
import rpy2.robjects as robjects

def execute_r_script(r_script_path, file_path):
    """
    Executes the R script and runs BrainWaveAnalysis for the given category.
    :param r_script_path: Path to the R script.
    :param file_path: Path to the directory containing CSV files.
    """
    try:
        print(f"Executing R script: {r_script_path}")
        robjects.r.source(r_script_path)  # Load the R script
        print("R script executed successfully.")
        
        # Access the BrainWaveAnalysis class from R
        BrainWaveAnalysis = robjects.globalenv['BrainWaveAnalysis']
        # Create an object of the BrainWaveAnalysis class
        analysis = BrainWaveAnalysis(file_path)
        analysis.plot_data()  # Plot the data
    except Exception as e:
        print(f"Error while executing R script: {e}")

def process_plotscode_directory(directory):
    """
    Automates processing of all Python-R file pairs in the plotscode directory.
    :param directory: Path to the 'plotscode' directory containing Python and R scripts.
    """
    # Iterate through all files in the directory
    categories = set()
    for file in os.listdir(directory):
        if file.endswith(".py"):  # Collect categories based on Python filenames
            category = os.path.splitext(file)[0]
            categories.add(category)

    for category in categories:
        print(f"Processing category: {category}")
        
        # Construct paths for the R script and file path
        r_script_path = os.path.join(directory, f"{category}.R")
        file_path = f"/Avatar/plots/plot_waves/brainwaves-csv/{category}"
        
        # Ensure the R script exists
        if os.path.exists(r_script_path):
            execute_r_script(r_script_path, file_path)
        else:
            print(f"Missing R script for category: {category}")

def main():
    # Directory containing Python and R scripts
    plotscode_directory = "/Avatar/plotscode"
    
    # Process all categories
    process_plotscode_directory(plotscode_directory)

if __name__ == "__main__":
    main()
