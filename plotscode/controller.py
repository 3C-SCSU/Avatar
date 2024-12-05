import os
import rpy2.robjects as robjects

class Controller:
    def __init__(self, r_script_path: str):
        """
        Initialize the Controller with categories and base paths.
        :param r_script_path: The file path of the R script
        """
        self.r_script_path = r_script_path

    def execute_r_script(self):
        """
        Executes the R script that processes all categories.
        """
        if not os.path.exists(self.r_script_path):
            print(f"R script not found: {self.r_script_path}")
            return

        try:
            print(f"Executing R script: {self.r_script_path}")
            robjects.r.source(self.r_script_path)  # Run the R script
            print("R script executed successfully.")
        except Exception as e:
            print(f"Error executing R script {self.r_script_path}: {e}")

# Example usage
if __name__ == "__main__":
    r_script_path = "./controller.R"  # File path to R script

    controller = Controller(r_script_path)

    # Execute the R script that processes all categories
    controller.execute_r_script()
