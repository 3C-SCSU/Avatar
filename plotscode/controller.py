import os

import rpy2.robjects as robjects


class Controller:
    def __init__(self, r_script_path: str):
        """
        Initialize the Controller with categories and base paths.
        :param r_script_path: The file path of the R script
        """
        self.r_script_path = r_script_path

    def execute_r_script(self, dataset_type=None):
        """
        Executes the R script that processes all categories.
        :param dataset_type: Optional parameter to specify which dataset to process ('rollback', 'refresh', or None for both)
        """
        if not os.path.exists(self.r_script_path):
            print(f"R script not found: {self.r_script_path}")
            return

        try:
            print(f"Executing R script: {self.r_script_path}")

            # Create datasets directory parameter
            if dataset_type:
                dataset_type = dataset_type.lower()
                if dataset_type not in ["rollback", "refresh"]:
                    print(f"Invalid dataset type: {dataset_type}. Using both datasets.")
                    dataset_type = None

            # Set up dataset parameter for R script
            if dataset_type:
                robjects.r(f'dataset_type <- "{dataset_type}"')
                print(f"Processing dataset: {dataset_type}")
            else:
                robjects.r("dataset_type <- NULL")
                print("Processing both datasets")

            # Run the R script
            robjects.r.source(self.r_script_path)
            print("R script executed successfully.")
        except Exception as e:
            print(f"Error executing R script {self.r_script_path}: {e}")


# Example usage
if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build an absolute path to the R script
    r_script_path = os.path.join(script_dir, "controller.R")

    print(f"Using R script path: {r_script_path}")
    controller = Controller(r_script_path)

    # Check if command line arguments were provided
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ["rollback", "refresh"]:
        # Execute the R script for the specific dataset
        controller.execute_r_script(sys.argv[1])
    else:
        # Execute the R script that processes all categories for both datasets
        controller.execute_r_script()
