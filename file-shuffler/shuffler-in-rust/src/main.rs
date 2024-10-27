/// File Shuffler
///
/// This application shuffles files in a specified source directory at defined intervals.
///
/// # Author
/// Joshua Olaoye

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
use clap::{Arg, Command};

/// Enum representing the time intervals for processing files.
///
/// This enum defines the various intervals at which the application can process files in the specified source directory.
///
/// # Variants
///
/// - `Never`: 
///   Indicates that the processing should occur only once and not repeat.
/// 
/// - `EveryWeek`: 
///   Indicates that the processing should occur once every week.
/// 
/// - `Every30Seconds`: 
///   Indicates that the processing should occur every 30 seconds.
#[derive(Debug)]
enum Interval {
    Never,
    EveryWeek,
    Every30Seconds,
}

/// Implementation of `FromStr` trait for `Interval` enum.
///
/// This implementation allows for the creation of an `Interval` instance from a string representation.
///
/// # Errors
///
/// Returns an error if the input string does not match one of the defined interval options:
/// - "0" for `Interval::Never`
/// - "1" for `Interval::EveryWeek`
/// - "2" for `Interval::Every30Seconds`
///
/// # Examples
///
/// ```
/// use std::str::FromStr;
///
/// let interval = Interval::from_str("1").unwrap(); // Converts to `Interval::EveryWeek`
/// let invalid_interval = Interval::from_str("3"); // Returns an error
/// ```
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

/// Generates a random timestamp within the last 10 days from the current time.
///
/// This function calculates a random offset in seconds, ranging from 0 to the total 
/// number of seconds in 10 days (864,000 seconds). It subtracts this offset from the 
/// current system time to produce a timestamp that is guaranteed to be within the last 
/// 10 days.
///
/// # Returns
///
/// A `SystemTime` representing a random point in time within the last 10 days.
///
/// # Examples
///
/// ```
/// let random_time = random_timestamp();
/// println!("Random timestamp: {:?}", random_time);
/// ```
fn random_timestamp() -> SystemTime {
    let now = SystemTime::now();
    let ten_days = Duration::from_secs(10 * 24 * 60 * 60); // 10 days in seconds
    let random_offset = rand::thread_rng().gen_range(0..=ten_days.as_secs());
    now - Duration::from_secs(random_offset)
}

/// Renames all files in the specified directory by moving them to a temporary directory 
/// with a unique random name. Hidden files (starting with '.') are removed.
///
/// # Parameters
///
/// - `dir`: A reference to a `Path` representing the directory containing the files to rename.
///
/// # Returns
///
/// Returns `Ok(())` on success, indicating that the files have been renamed and moved back 
/// to the original directory. Otherwise, it returns an `io::Result` indicating the type of 
/// error encountered.
///
/// # Errors
///
/// This function may return errors in the following cases:
/// 
/// - [`std::io::Error`]: If an I/O operation fails, such as when creating the temporary 
///   directory, reading from the directory, or renaming files.
/// 
/// - The function will remove hidden files (e.g., `.DS_Store` on macOS) but does not 
///   return errors for this operation; it simply skips these files.
///
/// # Note
///
/// After renaming, the function attempts to set the modified timestamp of the renamed files 
/// using a random timestamp generated within the last 10 days. If needed, this logic can be 
/// uncommented and implemented as per your requirements.
///
/// # Example
///
/// ```rust
/// let path = Path::new("/path/to/directory");
/// rename_files_in_directory(&path).expect("Failed to rename files");
/// ```
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

/// Checks if a given directory contains any subdirectories.
///
/// This function attempts to read the contents of the specified directory and 
/// determines if any of the entries are subdirectories. It iterates through 
/// the directory entries and returns `true` if at least one subdirectory is found; 
/// otherwise, it returns `false`.
///
/// # Parameters
///
/// - `dir`: A reference to a `Path` representing the directory to be checked.
///
/// # Returns
///
/// - `true` if the directory contains at least one subdirectory, 
/// - `false` if there are no subdirectories or if the directory cannot be read.
///
/// # Examples
///
/// ```
/// let path = Path::new("/some/directory");
/// if has_subdirectories(&path) {
///     println!("The directory contains subdirectories.");
/// } else {
///     println!("No subdirectories found.");
/// }
/// ```
fn has_subdirectories(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|mut entries| entries.any(|e| e.map(|e| e.path().is_dir()).unwrap_or(false)))
        .unwrap_or(false)
}

/// Moves all files from a specified directory to its parent directory and deletes the original directory.
///
/// This function iterates through all entries in the specified directory and moves any files 
/// it finds to the parent directory. If a file with the same name already exists in the parent 
/// directory, a unique name is generated by appending "_copyN" to the original filename, where 
/// N is a counter that increments until a unique name is found. Once all files have been moved, 
/// the original directory is deleted.
///
/// # Parameters
///
/// - `dir`: A reference to a `Path` representing the directory containing files to be moved.
///
/// # Returns
///
/// This function returns an `io::Result<()>`, which will be:
/// - `Ok(())` if all files were successfully moved and the directory was deleted,
/// - An `io::Error` if an error occurs during reading, moving files, or deleting the directory.
///
/// # Examples
///
/// ```
/// let path = Path::new("/some/directory");
/// if let Err(e) = move_files_to_parent_and_delete(&path) {
///     eprintln!("Failed to move files: {}", e);
/// }
/// ```
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

/// Recursively traverses directories, renaming files and managing directory structures.
///
/// This function takes a directory and processes it recursively. It first explores all immediate
/// subdirectories, invoking itself on each one. After processing all subdirectories, it checks
/// if the current directory is a leaf (i.e., it contains no further subdirectories).
///
/// If the directory is a leaf and its parent is the specified root directory, the function
/// renames the files within it. If the directory is not a leaf, it moves its files to the parent
/// directory and deletes the now-empty directory.
///
/// # Parameters
///
/// - `dir`: A reference to a `Path` representing the current directory being processed.
/// - `root_dir`: A reference to a `Path` representing the root directory from which the traversal
///   starts.
///
/// # Returns
///
/// This function returns an `io::Result<()>`, which will be:
/// - `Ok(())` if the processing is successful,
/// - An `io::Error` if an error occurs during directory traversal, file renaming, or moving files.
///
/// # Examples
///
/// ```
/// let root_path = Path::new("/some/root/directory");
/// if let Err(e) = process_directory_recursive(root_path, root_path) {
///     eprintln!("Error processing directory: {}", e);
/// }
/// ```
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

/// Checks if a directory is at least two levels deep.
///
/// A directory is considered "at least two levels deep" if it contains subdirectories, indicating 
/// that there is at least one level of nesting within it. This function essentially verifies 
/// whether the specified directory has any subdirectories.
///
/// # Parameters
///
/// - `dir`: A reference to a `Path` representing the directory to check.
///
/// # Returns
///
/// This function returns `true` if the directory contains subdirectories, indicating that it is 
/// at least two levels deep; otherwise, it returns `false`.
///
/// # Examples
///
/// ```
/// let path = Path::new("/some/directory");
/// if is_at_least_two_levels_deep(&path) {
///     println!("The directory is at least two levels deep.");
/// } else {
///     println!("The directory is not at least two levels deep.");
/// }
/// ```
fn is_at_least_two_levels_deep(dir: &Path) -> bool {
    return has_subdirectories(dir);
}

/// Prints a horizontal line of dashes in the terminal.
///
/// This function determines the terminal's width and prints a line of dashes
/// (`-`) that spans the entire width. If the terminal size cannot be determined,
/// it defaults to a width of 80 characters. An optional newline can be printed
/// after the line of dashes based on the `add_new_line` parameter.
///
/// # Parameters
///
/// - `add_new_line`: If set to `true`, an additional newline is printed after
///   the line of dashes. If set to `false`, no extra newline is printed.
///
/// # Example
///
/// ```
/// print_dashes(true); // Prints a line of dashes followed by a newline.
/// print_dashes(false); // Prints a line of dashes without an extra newline.
/// ```
///
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

/// Retrieves the formatted last modified time of a file.
///
/// This function takes a file path and returns the last modified time formatted
/// as a string. It retrieves the file's metadata to access the modification time,
/// converts it to a Unix timestamp, and formats it for better readability.
///
/// # Parameters
///
/// - `path`: A reference to a `PathBuf` representing the file whose modified time
///   is to be retrieved.
///
/// # Returns
///
/// Returns an `std::io::Result<String>` which:
/// - Contains the formatted date string on success.
/// - Returns an error if the file's metadata cannot be accessed or if the
///   modification time cannot be retrieved.
///
/// # Example
///
/// ```
/// let path = PathBuf::from("example.txt");
/// match get_formatted_modified_time(&path) {
///     Ok(formatted_time) => println!("Last modified: {}", formatted_time),
///     Err(e) => eprintln!("Error retrieving modified time: {}", e),
/// }
/// ```
///
fn get_formatted_modified_time(path: &PathBuf) -> std::io::Result<String> {
    let metadata = fs::metadata(path)?;
    let modified_time = metadata.modified()?;
    
    let timestamp = modified_time_to_unix(&modified_time)?;
    let formatted_date = format_timestamp(timestamp);
    
    Ok(formatted_date)
}

/// Converts a `SystemTime` object representing a modified time to a UNIX timestamp.
///
/// This function takes a reference to a `SystemTime` instance and calculates
/// the duration since the UNIX epoch (January 1, 1970). It returns the number of
/// seconds since that epoch as a `u64` value.
///
/// # Parameters
///
/// - `modified_time`: A reference to a `SystemTime` instance that represents
///   the time to be converted.
///
/// # Returns
///
/// Returns a `std::io::Result<u64>` which:
/// - Contains the UNIX timestamp in seconds on success.
/// - Returns an error if the provided `SystemTime` is earlier than the UNIX epoch.
///
/// # Panics
///
/// This function will panic if the provided `SystemTime` is in the past relative to
/// the UNIX epoch. To avoid this, ensure that the `modified_time` is a valid
/// `SystemTime` that is not earlier than January 1, 1970.
///
/// # Example
///
/// ```
/// use std::time::{SystemTime, UNIX_EPOCH};
///
/// let modified_time = SystemTime::now(); // Current time
/// match modified_time_to_unix(&modified_time) {
///     Ok(timestamp) => println!("UNIX Timestamp: {}", timestamp),
///     Err(e) => eprintln!("Error converting to UNIX timestamp: {}", e),
/// }
/// ```
///
fn modified_time_to_unix(modified_time: &SystemTime) -> std::io::Result<u64> {
    let duration_since_epoch = modified_time.duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    Ok(duration_since_epoch.as_secs())
}

/// Formats a UNIX timestamp into a human-readable string representation.
///
/// This function takes a UNIX timestamp (in seconds) and converts it into a
/// formatted string that represents the date and time in the Central Daylight Time
/// (CDT) zone (UTC-5). The output format is "Month Day, Year at Hour:Minute AM/PM".
///
/// # Parameters
///
/// - `timestamp`: A `u64` representing the UNIX timestamp (number of seconds
///   since January 1, 1970).
///
/// # Returns
///
/// Returns a `String` that contains the formatted date and time. If the provided
/// timestamp results in an invalid date, it returns "Invalid date".
///
/// # Example
///
/// ```
/// let timestamp = 1633072800; // Example UNIX timestamp
/// let formatted_date = format_timestamp(timestamp);
/// println!("Formatted date: {}", formatted_date);
/// ```
///
/// # Note
///
/// The function assumes that the input timestamp is in the UTC timezone and formats
/// it according to the Central Daylight Time (CDT) offset. Make sure to account
/// for daylight saving time changes as needed.
///
/// # Panics
///
/// This function will panic if the fixed offset cannot be created (which is unlikely
/// in practice).
///
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

/// Extracts a `DateTime<FixedOffset>` from a `LocalResult<DateTime<FixedOffset>>`.
///
/// This function handles different cases of `LocalResult` and returns the appropriate
/// `DateTime<FixedOffset>`. If the `LocalResult` is `None`, it returns `None`. If it is
/// `Single`, it returns the contained `DateTime`. If it is `Ambiguous`, it returns one of
/// the potential `DateTime` values, allowing for further handling if necessary.
///
/// # Parameters
///
/// - `local_result`: A `LocalResult<DateTime<FixedOffset>>` that may contain a single
///   `DateTime`, an ambiguous result, or no value at all.
///
/// # Returns
///
/// Returns an `Option<DateTime<FixedOffset>>`. This will be:
/// - `Some(DateTime<FixedOffset>)` if the input is `Single` or `Ambiguous` (with the first
///   `DateTime` from `Ambiguous`).
/// - `None` if the input is `None`.
///
/// # Example
///
/// ```
/// let result: LocalResult<DateTime<FixedOffset>> = /* some result */;
/// if let Some(dt) = extract_datetime(result) {
///     println!("Extracted date and time: {}", dt);
/// } else {
///     println!("No valid date and time found.");
/// }
/// ```
///
/// # Note
///
/// In the case of ambiguous results, the function currently returns the first `DateTime`
/// provided.
fn extract_datetime(local_result: LocalResult<DateTime<FixedOffset>>) -> Option<DateTime<FixedOffset>> {
    match local_result {
        LocalResult::Single(dt) => Some(dt),
        LocalResult::Ambiguous(dt1, _) => Some(dt1),
        LocalResult::None => None,
    }
}

/// The main entry point of the application.
///
/// This program shuffles files in a specified source directory at defined intervals.
/// It can run continuously based on the specified interval or process the directory just once.
///
/// # Usage
///
/// To run the application, provide a source directory and an optional interval flag:
///
/// ```bash
/// cargo run [--] <source_directory> [-i <interval>]
/// ```
///
/// # Arguments
///
/// - `<source_directory>`: 
///   The path to the directory containing files to be processed. This argument is required.
///
/// - `-i <interval>` or `--interval <interval>`: 
///   The time interval for processing the directory. Acceptable values are:
///     - `0`: Never repeat (process only once).
///     - `1`: Process every week.
///     - `2`: Process every 30 seconds.
///   If not provided, the default value is `0`.
///
/// # Example
///
/// To process files in the `example_dir` every 30 seconds:
///
/// ```bash
/// cargo run example_dir -i 2
/// ```
///
/// To process files in `example_dir` only once:
///
/// ```bash
/// cargo run example_dir
/// ```
///
/// To process files in `example_dir` every week:
///
/// ```bash
/// cargo run example_dir -i 1
/// ```
///
/// # Errors
///
/// This application will exit with an error message if:
/// - The provided source directory does not exist or is not a directory.
/// - The source directory is not at least two levels deep.
/// - The specified interval is not supported (values other than `0`, `1`, or `2`).
///
/// # Notes
/// If the specified interval is not supported, the application will use `0` (never) by default.
fn main() -> io::Result<()> {
    // Set up command-line argument parsing
    let matches = Command::new("File Shuffler")
        .version("1.0")
        .author("Your Name")
        .about("Shuffles files in a directory")
        .arg(Arg::new("source_directory")
            .help("The source directory to process")
            .required(true)
            .index(1))
        .arg(Arg::new("interval")
            .short('i') // Define short flag '-i'
            .long("interval") // Define long flag '--interval'
            .help("Set the processing interval (0: never, 1: every week, 2: every 30 seconds)")
            .default_value("0") // Default to "0" for 'never'
        ).get_matches();

    let source_dir = PathBuf::from(matches.get_one::<String>("source_directory").unwrap());
    let interval: Interval = matches.get_one::<String>("interval")
        .unwrap()
        .parse()
        .unwrap_or(Interval::Never); // Default to Never if invalid

    if !source_dir.exists() || !source_dir.is_dir() {
        eprintln!("Source directory not found or not a directory.");
        std::process::exit(1);
    }

    if !is_at_least_two_levels_deep(&source_dir) {
        eprintln!("Source directory is not at least 1 levels deep.");
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
