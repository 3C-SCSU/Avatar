use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use rand::Rng;
use filetime::{FileTime, set_file_times};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;
use std::str::FromStr;
use std::collections::HashSet;
use terminal_size::{Width, terminal_size};
use chrono::{DateTime, FixedOffset, TimeZone, LocalResult};

// Enum class for the time interval
#[derive(Debug)]
enum Interval {
    Never,
    EveryWeek,
    Every30Seconds,
}


impl FromStr for Interval {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(Interval::Never),
            "1" => Ok(Interval::EveryWeek),
            "2" => Ok(Interval::Every30Seconds),
            _ => Err(()),
        }
    }
}

/// Generates a random timestamp within the last 10 days.
fn random_timestamp() -> SystemTime {
    let now = SystemTime::now();
    let ten_days = Duration::from_secs(10 * 24 * 60 * 60); // 10 days in seconds
    let random_offset = rand::thread_rng().gen_range(0..=ten_days.as_secs());
    now - Duration::from_secs(random_offset)
}

/// Renames files in a given directory sequentially
fn rename_files_in_directory(dir: &Path) -> io::Result<()> {
    let mut used_numbers = HashSet::new();
    let mut rng = rand::thread_rng();

    // Create a temporary directory
    let temp_dir = dir.join("temp");
    fs::create_dir_all(&temp_dir)?;

    // Count the number of files in the directory
    let n = fs::read_dir(dir)?.count();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Check for hidden files and delete them (.DS_Store on Mac)
        if path.file_name().unwrap().to_str().unwrap().starts_with('.') {
            fs::remove_file(&path)?;
            continue; // Skip processing further for this entry
        }

        if path.is_file() {
            // Get last modified timestamp
            let timestamp = get_formatted_modified_time(&path)?;

            println!("| Original file\n| File name: {:?}\n| Timestamp: {:?}", path, timestamp);
            println!("•");
            // Generate a unique random number within the range of 1 to n
            let mut random_number;
            let mut new_path;
            loop {
                random_number = rng.gen_range(1..=n);
                new_path = temp_dir.join(format!("file_{}.txt", random_number));
                if !used_numbers.contains(&random_number) && !new_path.exists() {
                    used_numbers.insert(random_number);
                    break;
                }
            }

            // Move the file to the temporary directory with the new name
            fs::rename(&path, &new_path)?;

            // Uncomment and implement the timestamp update if needed
            let random_time = FileTime::from_system_time(random_timestamp());
            set_file_times(&new_path, random_time, random_time)?;

            let modified_timestamp = get_formatted_modified_time(&new_path)?;

            println!("| Modified file\n| File name: {:?}\n| Timestamp: {:?}\n", new_path, modified_timestamp);
        }
    }

    // Move files back from the temporary directory to the original directory
    for entry in fs::read_dir(&temp_dir)? {
        let entry = entry?;
        let temp_path = entry.path();
        let file_name = temp_path.file_name().unwrap();
        let final_path = dir.join(file_name);
        fs::rename(&temp_path, &final_path)?;
    }

    // Remove the temporary directory
    fs::remove_dir(&temp_dir)?;

    // FIXME: One off error on mac
    println!("Renamed {:?} file(s) in: {:?}", n, dir);
    print_dashes(true);
    Ok(())
}

/// Determines if a directory contains subdirectories.
fn has_subdirectories(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|mut entries| entries.any(|e| e.map(|e| e.path().is_dir()).unwrap_or(false)))
        .unwrap_or(false)
}

/// Moves files from a directory to its parent and deletes the directory.
fn move_files_to_parent_and_delete(dir: &Path) -> io::Result<()> {
    let parent_dir = dir.parent().unwrap();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let file_name = path.file_name().unwrap();
            let mut new_path = parent_dir.join(file_name);
        
            // Check for name conflicts and generate a unique name if necessary
            let mut counter = 1;
            while new_path.exists() {
                let new_file_name = format!("{}_copy{}", 
                    file_name.to_string_lossy(), // Original file name
                    counter // Append counter
                );
                new_path = parent_dir.join(new_file_name);
                counter += 1;
            }
        
            // Move the file to the parent directory
            fs::rename(&path, &new_path)?;
        }        
    }

    // Delete the now-empty directory
    fs::remove_dir(dir)?;
    Ok(())
}

/// Recursively traverses directories and renames files
fn process_directory_recursive(dir: &Path, root_dir: &Path) -> io::Result<()> {

    // Process only the immediate subdirectories
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        // Process only directories
        if path.is_dir() {
            process_directory_recursive(&path, &root_dir)?;
        }
    }

    // After processing subdirectories, check if the current directory is now a leaf
    if dir != root_dir && !has_subdirectories(dir) {
        if dir.parent() == Some(root_dir) {
            print_dashes(false);
            println!("Currently processing {:?}\n", dir);
            rename_files_in_directory(dir)?;
            return Ok(())
        }
        move_files_to_parent_and_delete(dir)?;
    }

    Ok(())
}

fn is_at_least_two_levels_deep(dir: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_dir() {
                    return true;
                }
            }
        }
    }
    false
}

fn print_dashes(add_new_line: bool) {
    let width = if let Some((Width(w), _)) = terminal_size() {
        w as usize
    } else {
        80 // Default width if terminal size can't be determined
    };

    println!("{}", "-".repeat(width));

    if add_new_line {
        println!(); // Print an additional new line if specified
    }
}

/// Function to get the formatted last modified time of a file
fn get_formatted_modified_time(path: &PathBuf) -> std::io::Result<String> {
    let metadata = fs::metadata(path)?;
    let modified_time = metadata.modified()?;
    
    let timestamp = modified_time_to_unix(&modified_time)?;
    let formatted_date = format_timestamp(timestamp);
    
    Ok(formatted_date)
}

/// Function to convert SystemTime to UNIX timestamp
fn modified_time_to_unix(modified_time: &SystemTime) -> std::io::Result<u64> {
    let duration_since_epoch = modified_time.duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    Ok(duration_since_epoch.as_secs())
}

/// Function to format UNIX timestamp to a human-readable string
fn format_timestamp(timestamp: u64) -> String {
    // Define a fixed offset for CDT (UTC-5)
    let fixed_offset = FixedOffset::west_opt(5 * 3600).expect("Invalid offset"); // 5 hours behind UTC

    // Convert the timestamp to DateTime<FixedOffset>
    let datetime: LocalResult<DateTime<FixedOffset>> = fixed_offset.timestamp_opt(timestamp as i64, 0);

    // Extract the DateTime
    match extract_datetime(datetime) {
        Some(dt) => dt.format("%B %d, %Y at %I:%M %p").to_string(),
        None => "Invalid date".to_string(), // Handle the None case as needed
    }
}

fn extract_datetime(local_result: LocalResult<DateTime<FixedOffset>>) -> Option<DateTime<FixedOffset>> {
    match local_result {
        LocalResult::Single(dt) => Some(dt),
        LocalResult::Ambiguous(dt1, _) => Some(dt1), // You can choose which one to return or handle both
        LocalResult::None => None,
    }
}

fn main() -> io::Result<()> {
    // Get the source directory and interval from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <source_directory> <interval>", args[0]);
        eprintln!("Interval: 0 for never, 1 for every week, 2 for every 30 seconds");
        std::process::exit(1);
    }

    let source_dir = PathBuf::from(&args[1]);
    let interval: Interval = args[2].parse().unwrap_or(Interval::Never);

    if !source_dir.exists() || !source_dir.is_dir() {
        eprintln!("Source directory not found or not a directory.");
        std::process::exit(1);
    }

    if !is_at_least_two_levels_deep(&source_dir) {
        eprintln!("Source directory is not at least 2 levels deep.");
        std::process::exit(1);
    }

    match interval {
        Interval::Never => {
            process_directory_recursive(&source_dir, &source_dir)?;
        }
        Interval::EveryWeek => loop {
            process_directory_recursive(&source_dir, &source_dir)?;
            thread::sleep(Duration::from_secs(7 * 24 * 60 * 60)); // Sleep for 1 week
        },
        Interval::Every30Seconds => loop {
            process_directory_recursive(&source_dir, &source_dir)?;
            thread::sleep(Duration::from_secs(30)); // Sleep for 30 seconds
        },
    }

    Ok(())
}


// Cargo
// Cargo run <dirname> <time interval>
// 0 - Never
// 1 - Every week
// 2 - Every 30 seconds