import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

def main():
    # Path to your R script
    r_script_path = "/Avatar/plots/land.R"

    # script path for land : /Avatar/plots/land.R

        try:
        # Load the R script
        print(f"Executing R script: {r_script_path}")
        robjects.r.source(r_script_path)
        print("R script executed successfully.")
    except Exception as e:
        print(f"An error occurred while executing the R script: {e}")
        return
    
    # Access the BrainWaveAnalysis class from R
    BrainWaveAnalysis = robjects.globalenv['BrainWaveAnalysis']
    
    # Create an object of the BrainWaveAnalysis class
    file_path = "/Avatar/plots/plot_waves/brainwaves-csv/land"
    analysis = BrainWaveAnalysis(file_path)

    # file path for land : /Avatar/plots/plot_waves/brainwaves-csv/land

    # You can now access methods or attributes of the R6 class from Python, e.g.,:
    # For example, you can invoke the plot_data method:
    analysis.plot_data()

if __name__ == "__main__":
    main()

