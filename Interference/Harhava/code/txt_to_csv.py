import os
import csv
import glob


def convert_txt_to_csv(file_path):
    """
    Convert a TXT file to CSV format and remove first 7 lines

    Args:
        file_path: Path to the input TXT file
    """
    # Create output CSV file path (same name but with .csv extension)
    base_name = os.path.splitext(file_path)[0]
    csv_path = base_name + '.csv'

    # Read input file and skip first 7 lines
    with open(file_path, 'r') as txt_file:
        # Skip first 7 lines
        lines = txt_file.readlines()[7:]

        # Write to CSV file
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for line in lines:
                # Split the line by whitespace or another delimiter
                # You may need to adjust this depending on your text file format
                row = line.strip().split()
                if row:  # Only write non-empty rows
                    csv_writer.writerow(row)

    print(f"Converted {file_path} to {csv_path} (removed first 7 lines)")


def process_directory(directory_path):
    """
    Process all TXT files in the given directory

    Args:
        directory_path: Path to directory containing TXT files
    """
    # Find all .txt files in the directory
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    if not txt_files:
        print(f"No TXT files found in {directory_path}")
        return

    print(f"Found {len(txt_files)} TXT files to process")

    # Process each TXT file
    for txt_file in txt_files:
        convert_txt_to_csv(txt_file)

    print("All files processed successfully")


if __name__ == "__main__":

    directory_path = r'..\data\Mie_angles'
    process_directory(directory_path)