# Set the CRAN mirror globally
options(repos = c(CRAN = "https://cran.rstudio.com"))

# List of required packages
required_packages <- c(
  "R6",
  "tidyverse", 
  "readr",
  "janitor",
  "stringr"
)

# Function to check if a package is installed
install_if_missing <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    cat("Installing", package, "\n")
    install.packages(package)
  } else {
    cat(package, "is already installed.\n")
  }
}

# Install all the required packages
for (pkg in required_packages) {
  install_if_missing(pkg)
}

cat("All dependencies are installed!\n")
