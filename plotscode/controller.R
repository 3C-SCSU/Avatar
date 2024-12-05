# Import required libraries
library(R6)
library(tidyverse)
library(readr)
library(janitor)
library(stringr)

# Define the BrainWaveAnalysis R6 class
BrainWaveAnalysis <- R6Class("BrainWaveAnalysis",
  public = list(
    categories = NULL,       # List of categories to process
    base_path = NULL,        # Base file path for data
    results = list(),        # Store analysis results for each category
    
    # Constructor to initialize the object with categories and base path
    initialize = function(categories, base_path) {
      self$categories = categories
      self$base_path = base_path
    },
    
    # Method to process all categories
    process_all_categories = function() {
      for (category in self$categories) {
        cat(sprintf("Processing category: %s\n", category))
        category_path <- file.path(self$base_path, category)
        if (!dir.exists(category_path)) {
          cat(sprintf("Directory not found: %s\n", category_path))
          next
        }
        
        # Perform analysis for the category
        tryCatch({
          result <- self$analyze_category(category_path)
          self$results[[category]] <- result
          cat(sprintf("Successfully processed: %s\n", category))
        }, error = function(e) {
          cat(sprintf("Error processing category %s: %s\n", category, e$message))
        })
      }
    },
    
    # Method to analyze a single category
    analyze_category = function(file_path) {
      # Load data files
      list_of_files <- list.files(
        path = file_path,
        recursive = TRUE,
        pattern = "\\.csv$",
        full.names = TRUE
      )
      
      if (length(list_of_files) == 0) {
        stop("No CSV files found in directory.")
      }
      
      df <- readr::read_csv(list_of_files)
      df <- df %>% clean_names()
      
      # Reshape and process data
      df_long <- df %>%
        pivot_longer(
          cols = starts_with("exg"),
          names_to = "Channel",
          values_to = "Amplitude"
        ) %>%
        mutate(ChannelNumber = str_pad(
          as.integer(str_extract(Channel, "\\d+")) + 1, 
          width = 2, 
          pad = "0"
        ))
      
      # Bin data
      df_binned <- df_long %>%
        group_by(ChannelNumber) %>%
        mutate(Bin = floor(sample_index / 88) * 88) %>%
        group_by(ChannelNumber, Bin) %>%
        summarise(
          AmplitudeMean = mean(Amplitude, na.rm = TRUE),
          .groups = 'drop'
        )
      
      # Subset for plotting
      df_subset <- df_long %>%
        group_by(ChannelNumber) %>%
        mutate(Row = row_number()) %>%
        filter(Row %% 120 == 0)
      
      # Fit polynomial model
      model <- lm(Amplitude ~ poly(sample_index, 12), data = df_subset)
      
      # Generate predictions
      min_index <- min(df_subset$sample_index)
      max_index <- max(df_subset$sample_index)
      xdf <- data.frame(sample_index = seq(min_index, max_index, length.out = 300))
      xdf$AmplitudeFitted <- predict(model, newdata = xdf)
      
      # Plot data
      self$plot_data(df_subset, xdf, category = basename(file_path))
      
      # Return results
      list(
        raw_data = df,
        long_data = df_long,
        binned_data = df_binned,
        subset_data = df_subset,
        model = model,
        predictions = xdf
      )
    },
    
    # Plot the data for a category
    plot_data = function(df_subset, xdf, category) {
      # Construct a file name based on the category, ensuring the plot is saved in the 'plots' directory
      pdf_filename <- file.path("plots", paste0(category, "_plots.pdf"))
      
      # Check if the 'plots' directory exists, if not, create it
      if (!dir.exists("plots")) {
        dir.create("plots")
      }

      # Open a PDF graphics device with the custom name
      pdf(file = pdf_filename, width = 8, height = 6)
      
      p <- ggplot(df_subset, aes(x = sample_index, y = Amplitude)) +
        geom_smooth(
          se = FALSE,
          method = "loess",
          span = 0.25,
          aes(group = ChannelNumber, color = ChannelNumber),
          linewidth = 1
        ) +
        geom_line(
          data = xdf,
          aes(x = sample_index, y = AmplitudeFitted),
          linewidth = 5,
          color = "red",
          linetype = "twodash"
        ) +
        theme_minimal() +
        labs(
          title = sprintf("Brain Wave Signals: %s", category),
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

      dev.off()
    }
  )
)

# Example usage
categories <- c("backward", "forward", "land", "left", "right", "takeoff")
base_path <- # Path to the brainwave-csv directory

analysis <- BrainWaveAnalysis$new(categories, base_path)
analysis$process_all_categories()
