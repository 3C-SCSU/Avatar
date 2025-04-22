# Import required libraries
library(R6)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(janitor)
library(stringr)

# If dataset_type is not defined, set it to NULL (process both datasets)
if (!exists("dataset_type")) {
  dataset_type <- NULL
}

# Define the datasets to process based on dataset_type parameter
datasets <- if(is.null(dataset_type)) {
  c("rollback", "refresh")
} else {
  c(dataset_type)
}

# Define the BrainWaveAnalysis R6 class
BrainWaveAnalysis <- R6Class("BrainWaveAnalysis",
  public = list(
    categories = NULL,       # List of categories to process
    base_path = NULL,        # Base file path for data
    output_dir = NULL,       # Output directory for plots
    results = list(),        # Store analysis results for each category
    
    # Constructor to initialize the object with categories and base path
    initialize = function(categories, base_path, output_dir) {
      self$categories = categories
      self$base_path = base_path
      self$output_dir = output_dir
      
      # Create the output directory if it doesn't exist
      if (!dir.exists(self$output_dir)) {
        dir.create(self$output_dir, recursive = TRUE)
      }
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
          # If there's an error, try to create a minimal plot anyway to ensure the PDF exists
          self$create_placeholder_plot(category)
        })
      }
    },
    
    # Create a simple placeholder plot if normal processing fails
    create_placeholder_plot = function(category) {
      cat(sprintf("Creating placeholder plot for category: %s\n", category))
      
      # Generate a simple dataset
      sample_index <- 1:300
      amplitude <- runif(300, -30000, 30000)
      df_simple <- data.frame(
        sample_index = sample_index,
        Amplitude = amplitude,
        ChannelNumber = rep("01", 300)
      )
      
      # Generate simple predictions
      xdf <- data.frame(
        sample_index = seq(min(sample_index), max(sample_index), length.out = 300),
        AmplitudeFitted = sin(seq(0, 6*pi, length.out = 300)) * 20000
      )
      
      # Create a simple plot
      pdf_filename <- file.path(self$output_dir, paste0(category, "_plots.pdf"))
      pdf(file = pdf_filename, width = 8, height = 6)
      
      p <- ggplot(df_simple, aes(x = sample_index, y = Amplitude)) +
        geom_line(color = "blue") +
        geom_line(
          data = xdf,
          aes(x = sample_index, y = AmplitudeFitted),
          linewidth = 2,
          color = "red",
          linetype = "twodash"
        ) +
        theme_minimal() +
        labs(
          title = sprintf("Brain Wave Signals: %s (Placeholder)", category),
          x = "Number of Samples",
          y = "EEG Signal Amplitude",
          subtitle = "Generated as a placeholder due to data processing error"
        ) +
        coord_cartesian(ylim = c(-30000, 30000), xlim = c(-1, 300)) +
        theme(
          plot.title = element_text(hjust = 0.5),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
          plot.margin = margin(10, 10, 10, 10)
        )
      
      print(p)
      dev.off()
      
      cat(sprintf("Created placeholder plot: %s\n", pdf_filename))
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
      
      # Check dataset type to determine how to read CSV files
      dataset_name <- basename(dirname(file_path))
      cat(sprintf("Processing files from dataset: %s\n", dataset_name))
      
      # Try to read files with appropriate format
      if (dataset_name == "rollback") {
        # Rollback dataset: comma-separated with headers
        cat("Reading rollback format: comma-separated with headers\n")
        df_list <- lapply(list_of_files, function(file) {
          tryCatch({
            readr::read_csv(file, show_col_types = FALSE)
          }, error = function(e) {
            cat(sprintf("  Error reading file %s: %s\n", file, e$message))
            return(NULL)
          })
        })
        
        # Remove NULL entries (failed reads)
        df_list <- df_list[!sapply(df_list, is.null)]
        
        # Combine all data frames
        if (length(df_list) > 0) {
          # Check if there are any data frames to combine
          if (length(df_list) == 1) {
            df <- df_list[[1]]
          } else {
            # When combining multiple data frames, ensure consistent types
            df <- tryCatch({
              bind_rows(df_list)
            }, error = function(e) {
              cat(sprintf("  Error binding rows: %s\n", e$message))
              cat("  Trying alternative method to combine data frames...\n")
              
              # Identify common columns across all data frames
              common_cols <- Reduce(intersect, lapply(df_list, names))
              cat(sprintf("  Common columns: %s\n", paste(common_cols, collapse=", ")))
              
              # Select only common columns and convert to character to avoid type conflicts
              combined_df <- NULL
              for (df_item in df_list) {
                # Convert all columns to character to avoid type conflicts
                for (col in common_cols) {
                  df_item[[col]] <- as.character(df_item[[col]])
                }
                df_subset <- df_item[, common_cols, drop=FALSE]
                
                if (is.null(combined_df)) {
                  combined_df <- df_subset
                } else {
                  combined_df <- rbind(combined_df, df_subset)
                }
              }
              return(combined_df)
            })
          }
        } else {
          stop("Failed to read any files")
        }
        
      } else if (dataset_name == "refresh") {
        # Refresh dataset: space-separated without headers
        cat("Reading refresh format: space-separated without headers\n")
        
        # Use custom column names for the refresh dataset
        # Based on examining the file structure, first column is sample_index, 
        # followed by EEG data columns
        col_names <- c("sample_index")
        
        # Add EEG channel columns (at least 8 channels observed in sample file)
        for (i in 1:8) {
          col_names <- c(col_names, paste0("eeg_", i))
        }
        
        # Add remaining columns with generic names
        for (i in 1:30) {  # Increased to 30 columns to handle more data
          col_names <- c(col_names, paste0("misc_", i))
        }
        
        # Read files one by one
        df_list <- lapply(list_of_files, function(file) {
          tryCatch({
            # Use read.table for more robust handling of inconsistent whitespace
            temp_df <- read.table(
              file,
              sep = "",  # Auto-detect separators (space, tab, etc.)
              header = FALSE,
              stringsAsFactors = FALSE,
              na.strings = c("NA", "", "NULL"),
              col.names = col_names,
              fill = TRUE,  # Fill rows with NA if they're short
              comment.char = ""  # Don't treat any character as comment
            )
            
            # Convert to tibble for consistency with other code
            as_tibble(temp_df)
          }, error = function(e) {
            cat(sprintf("  Error reading file %s: %s\n", file, e$message))
            return(NULL)
          })
        })
        
        # Remove NULL entries (failed reads)
        df_list <- df_list[!sapply(df_list, is.null)]
        
        # Combine all data frames
        if (length(df_list) > 0) {
          # When dealing with refresh dataset, we need to handle potential inconsistencies
          # Find common columns that exist in all dataframes
          if (length(df_list) == 1) {
            df <- df_list[[1]]
          } else {
            # Find common columns across all dataframes
            common_cols <- Reduce(intersect, lapply(df_list, names))
            cat(sprintf("  Common columns across all files: %s\n", 
                        paste(head(common_cols, 5), collapse=", "),
                        if(length(common_cols) > 5) "..." else ""))
            
            # Use only the sample_index and first 8 columns (which should be EEG data)
            selected_cols <- c("sample_index")
            for (col in common_cols) {
              if (grepl("^eeg_", col)) {
                selected_cols <- c(selected_cols, col)
              }
            }
            
            # If we didn't find any eeg_ columns, use the next 8 columns after sample_index
            if (length(selected_cols) <= 1) {
              cat("  No eeg_ columns found, using first 8 columns as EEG data\n")
              # Use the first 8 columns after sample_index
              all_cols <- unique(unlist(lapply(df_list, names)))
              all_cols <- all_cols[all_cols != "sample_index"]
              selected_cols <- c("sample_index", all_cols[1:min(8, length(all_cols))])
            }
            
            cat(sprintf("  Selected columns: %s\n", 
                        paste(head(selected_cols, 5), collapse=", "),
                        if(length(selected_cols) > 5) "..." else ""))
            
            # Combine data frames using only selected columns and ensuring numeric type
            df <- bind_rows(lapply(df_list, function(df_item) {
              # Ensure all selected columns exist
              for (col in selected_cols) {
                if (!col %in% names(df_item)) {
                  df_item[[col]] <- NA
                }
              }
              
              # Select only required columns
              df_subset <- df_item[, selected_cols, drop=FALSE]
              
              # Ensure sample_index is numeric
              if ("sample_index" %in% names(df_subset)) {
                df_subset$sample_index <- as.numeric(df_subset$sample_index)
              }
              
              # Ensure EEG columns are numeric
              for (col in selected_cols) {
                if (col != "sample_index") {
                  df_subset[[col]] <- as.numeric(df_subset[[col]])
                }
              }
              
              return(df_subset)
            }))
          }
        } else {
          stop("Failed to read any files")
        }
        
      } else {
        # Unknown dataset, try different approaches
        cat("Unknown dataset, trying different CSV formats\n")
        
        # First try standard CSV format
        tryCatch({
          df_list <- lapply(list_of_files, function(file) {
            tryCatch({
              readr::read_csv(file, show_col_types = FALSE)
            }, error = function(e) {
              return(NULL)
            })
          })
          
          # Remove NULL entries and combine
          df_list <- df_list[!sapply(df_list, is.null)]
          if (length(df_list) > 0) {
            if (length(df_list) == 1) {
              df <- df_list[[1]]
            } else {
              # Find common columns
              common_cols <- Reduce(intersect, lapply(df_list, names))
              
              # Combine using only common columns to avoid type conflicts
              df <- bind_rows(lapply(df_list, function(df_item) {
                # Ensure all common columns are character type to avoid type conflicts
                for (col in common_cols) {
                  df_item[[col]] <- as.character(df_item[[col]])
                }
                return(df_item[, common_cols, drop=FALSE])
              }))
            }
          } else {
            stop("Failed with standard CSV format")
          }
        }, error = function(e) {
          # If standard CSV format fails, try space-delimited
          cat("Trying space-delimited format\n")
          
          col_names <- c("sample_index")
          for (i in 1:8) {
            col_names <- c(col_names, paste0("eeg_", i))
          }
          for (i in 1:30) {  # Increased to 30 columns
            col_names <- c(col_names, paste0("misc_", i))
          }
          
          df_list <- lapply(list_of_files, function(file) {
            tryCatch({
              temp_df <- read.table(
                file,
                sep = "",
                header = FALSE,
                stringsAsFactors = FALSE,
                na.strings = c("NA", "", "NULL"),
                col.names = col_names,
                fill = TRUE,
                comment.char = ""
              )
              as_tibble(temp_df)
            }, error = function(e2) {
              return(NULL)
            })
          })
          
          # Remove NULL entries and combine
          df_list <- df_list[!sapply(df_list, is.null)]
          if (length(df_list) > 0) {
            # Same approach as refresh dataset
            if (length(df_list) == 1) {
              df <- df_list[[1]]
            } else {
              # Find common columns
              common_cols <- Reduce(intersect, lapply(df_list, names))
              
              # Select only required columns like sample_index and eeg columns
              selected_cols <- c("sample_index")
              for (col in common_cols) {
                if (grepl("^eeg_", col)) {
                  selected_cols <- c(selected_cols, col)
                }
              }
              
              # If no eeg columns found, use first 8 columns
              if (length(selected_cols) <= 1) {
                all_cols <- unique(unlist(lapply(df_list, names)))
                all_cols <- all_cols[all_cols != "sample_index"]
                selected_cols <- c("sample_index", all_cols[1:min(8, length(all_cols))])
              }
              
              # Combine with type conversion to ensure consistency
              df <- bind_rows(lapply(df_list, function(df_item) {
                # Ensure all selected columns exist
                for (col in selected_cols) {
                  if (!col %in% names(df_item)) {
                    df_item[[col]] <- NA
                  }
                }
                
                # Select only required columns
                df_subset <- df_item[, selected_cols, drop=FALSE]
                
                # Ensure sample_index is numeric
                if ("sample_index" %in% names(df_subset)) {
                  df_subset$sample_index <- as.numeric(df_subset$sample_index)
                }
                
                # Ensure EEG columns are numeric
                for (col in selected_cols) {
                  if (col != "sample_index") {
                    df_subset[[col]] <- as.numeric(df_subset[[col]])
                  }
                }
                
                return(df_subset)
              }))
            }
          } else {
            stop("Failed to read files with any method")
          }
        })
      }
      
      # Clean column names
      df <- df %>% clean_names()
      
      # Reshape and process data
      # Print column names for debugging
      cat("Available columns in dataset:", paste(names(df)[1:min(5, length(names(df)))], collapse=", "), 
          if(length(names(df)) > 5) "..." else "", "\n")
      
      # Identify EEG data columns
      eeg_cols <- NULL
      
      # Check for known EEG column patterns
      if (any(grepl("^exg_", names(df)))) {
        eeg_cols <- names(df)[grepl("^exg_", names(df))]
        cat("Found", length(eeg_cols), "columns with 'exg_' pattern\n")
      } else if (any(grepl("^exg_channel_", names(df)))) {
        eeg_cols <- names(df)[grepl("^exg_channel_", names(df))]
        cat("Found", length(eeg_cols), "columns with 'exg_channel_' pattern\n")
      } else if (any(grepl("^eeg_", names(df)))) {
        eeg_cols <- names(df)[grepl("^eeg_", names(df))]
        cat("Found", length(eeg_cols), "columns with 'eeg_' pattern\n")
      } else if (any(grepl("Channel", names(df), ignore.case = TRUE))) {
        eeg_cols <- names(df)[grepl("Channel", names(df), ignore.case = TRUE)]
        cat("Found", length(eeg_cols), "columns with 'Channel' pattern\n")
      }
      
      # If no EEG columns found with patterns, use numeric columns as fallback
      if (is.null(eeg_cols) || length(eeg_cols) == 0) {
        # Get all column names
        all_cols <- names(df)
        
        # Remove sample_index or similar
        all_cols <- all_cols[!grepl("^sample|^index|^time", all_cols, ignore.case = TRUE)]
        
        # Get the first 8 remaining columns (which should be EEG data in refresh dataset)
        eeg_cols <- all_cols[1:min(8, length(all_cols))]
        cat("Using", length(eeg_cols), "columns as EEG data\n")
        
        if (length(eeg_cols) == 0) {
          stop("No suitable EEG data columns found in the dataset")
        }
      }
      
      # Ensure sample_index column exists
      if (!"sample_index" %in% names(df)) {
        # Look for potential sample index columns
        potential_index_cols <- names(df)[grepl("^sample|^index|^time", names(df), ignore.case = TRUE)]
        
        if (length(potential_index_cols) > 0) {
          # Use the first match
          df <- df %>% rename_with(~ "sample_index", potential_index_cols[1])
        } else {
          # If no suitable column found, use the first column and assume it's sample_index
          df <- df %>% rename_with(~ "sample_index", 1)
        }
      }
      
      # Ensure data types
      df$sample_index <- as.numeric(df$sample_index)
      for (col in eeg_cols) {
        df[[col]] <- as.numeric(df[[col]])
      }
      
      # Reshape data
      cat("Reshaping data with EEG columns:", paste(eeg_cols[1:min(3, length(eeg_cols))], collapse=", "), 
          if(length(eeg_cols) > 3) "..." else "", "\n")
      
      df_long <- df %>%
        select(sample_index, all_of(eeg_cols)) %>%
        pivot_longer(
          cols = all_of(eeg_cols),
          names_to = "Channel",
          values_to = "Amplitude"
        ) %>%
        mutate(ChannelNumber = str_pad(
          # Extract channel number from various formats, default to "01" if not found
          ifelse(
            grepl("\\d+", Channel),
            as.integer(str_extract(Channel, "\\d+")) + 1,
            as.integer(factor(Channel)) # Use factor level as channel number if no number in name
          ),
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
      
      # Subset for plotting (use every 120th row to reduce data size)
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
      # Construct a file name based on the category, ensuring the plot is saved in the output directory
      pdf_filename <- file.path(self$output_dir, paste0(category, "_plots.pdf"))
      
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

# Categories to process
categories <- c("backward", "forward", "land", "left", "right", "takeoff")

# Process each dataset
for (dataset in datasets) {
  cat(sprintf("\n=== Processing %s dataset ===\n", dataset))
  
  # Base path for the current dataset
  base_path <- file.path("path to your data", dataset)
  # Like this: "/Users/arham/Documents/Projects/Avatar/data_csv"
  
  # Output directory for the plots
  output_dir <- file.path("path where you want to save the plots", dataset)
  # Like this: "/Users/arham/Documents/Projects/Avatar/plotscode/plots"
  # Create the analysis object
  analysis <- BrainWaveAnalysis$new(categories, base_path, output_dir)
  
  # Process all categories
  analysis$process_all_categories()
}
