# Import required libraries
library(R6)            # For object-oriented programming
library(tidyverse)     # For data manipulation and visualization
library(readr)         # For reading CSV files
library(janitor)       # For cleaning column names
library(stringr)       # For string manipulation

# Define the BrainWaveAnalysis R6 class
BrainWaveAnalysis <- R6Class("BrainWaveAnalysis",
  public = list(
    df = NULL,                   # Data frame to hold the raw data
    df_long = NULL,              # Long-format data
    df_binned = NULL,            # Binned data for plotting
    bin_size = 88,               # Default bin size
    model = NULL,                # Polynomial model
    df_subset = NULL,            # Subset for plotting
    xdf = NULL,                  # Data for prediction
    
    # Constructor to initialize the object with files
    initialize = function(file_path) {
      # Load the data
      list_of_files <- list.files(
        path = file_path,
        recursive = TRUE,
        pattern = "\\.csv$",
        full.names = TRUE
      )
      
      self$df <- readr::read_csv(list_of_files)
      self$clean_data()
    },
    
    # Method to clean column names
    clean_data = function() {
      self$df <- self$df %>% clean_names()
      self$reshape_data()
    },
    
    # Method to reshape data into long format
    reshape_data = function() {
      self$df_long <- self$df %>%
        pivot_longer(
          cols = starts_with("exg"),
          names_to = "Channel",
          values_to = "Amplitude"
        )
      
      # Modify the 'Channel' column to show numbers starting from 01 to 16
      self$df_long <- self$df_long %>%
        mutate(ChannelNumber = str_pad(as.integer(str_extract(Channel, "\\d+")) + 1, width = 2, pad = "0"))
      
      self$bin_data()
    },
    
    # Method to bin the data
    bin_data = function() {
      self$df_binned <- self$df_long %>%
        group_by(ChannelNumber) %>%
        mutate(Bin = floor(sample_index / self$bin_size) * self$bin_size) %>%
        group_by(ChannelNumber, Bin) %>%
        summarise(
          AmplitudeMean = mean(Amplitude, na.rm = TRUE),
          .groups = 'drop'
        )
      
      self$prepare_for_plotting()
    },
    
    # Prepare the data subset for plotting (every 120th point)
    prepare_for_plotting = function() {
      self$df_subset <- self$df_long %>%
        group_by(ChannelNumber) %>%
        mutate(Row = row_number()) %>%
        filter(Row %% 120 == 0)
      
      self$fit_model()
    },
    
    # Fit the polynomial model
    fit_model = function() {
      self$model <- lm(Amplitude ~ poly(sample_index, 12), data = self$df_subset)
      self$predict_data()
    },
    
    # Generate predictions
    predict_data = function() {
      min_index <- min(self$df_subset$sample_index)
      max_index <- max(self$df_subset$sample_index)
      
      self$xdf <- data.frame(
        sample_index = seq(min_index, max_index, length.out = 300)
      )
      
      self$xdf$AmplitudeFitted <- predict(self$model, newdata = self$xdf)
      self$plot_data()
    },
    
    # Plot the data using ggplot2
    plot_data = function() {
      dev.new()  # Open a new graphics device window
      
      p <- ggplot(self$df_subset, aes(x = sample_index, y = Amplitude)) +
        geom_smooth(
          se = FALSE,
          method = "loess",
          span = 0.25,
          aes(group = ChannelNumber, color = ChannelNumber),
          linewidth = 1
        ) +
        geom_line(
          data = self$xdf,
          aes(x = sample_index, y = AmplitudeFitted),
          linewidth = 5,
          color = "red",
          linetype = "twodash"
        ) +
        theme_minimal() +
        labs(
          title = "Brain Wave Signals: Backward",
          x = "Number of Samples",
          y = "EEG Signal Amplitude",
          color = "Channel"
        ) +
        guides(color = guide_legend(title = "Channel")) +
        coord_cartesian(ylim = c(-30000, 30000), xlim = c(-1, 300)) +
        theme(
          plot.title = element_text(hjust = 0.5),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
          plot.margin = margin(10, 10, 10, 10),
          legend.title = element_text(size = 10),
          legend.text = element_text(size = 9)
        )
      
      print(p)  # Display the plot
      readline(prompt = "Press Enter to close the plot...")  # Keep the plot open
    }
  )
)

# Create an object of the BrainWaveAnalysis class
analysis <- BrainWaveAnalysis$new("/Avatar/plots/plot_waves/brainwaves-csv/backward")

# file path for backward : /Avatar/plots/plot_waves/brainwaves-csv/backward
