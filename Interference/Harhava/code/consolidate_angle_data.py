import os
import glob
import pandas as pd
import numpy as np
import re


def extract_data_from_csv_files(folder_path):
    """
    Process all CSV files in the given folder with naming format {float}r-{color}
    and extract radius, wavelength, mean, and standard deviation data.
    Also processes reference measurements for each color.

    Args:
        folder_path: Path to folder containing CSV files
    """

    # List to store data for all files
    all_data = []

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # First, look for reference files and process them
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        base_name = os.path.splitext(filename)[0]

        # Check if this is a reference file for any color
        # Assuming reference files have names like "reference-red.csv", "reference-blue.csv", etc.

    # Now process the regular measurement files
    for csv_file in csv_files:
        # Extract filename without extension
        filename = os.path.basename(csv_file)
        base_name = os.path.splitext(filename)[0]

        # Extract radius and color using regex
        match = re.match(r'(0\.5|0\.75|1)r_(\d+)deg', base_name)
        if not match:
            print(f"Skipping file {filename} - doesn't match expected naming pattern")
            continue

        radius = float(match.group(1))
        angle = int(match.group(2))# Assuming angle is in degrees

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if the required column exists
            if 'Ch0[V]' not in df.columns:
                print(f"Warning: File {filename} doesn't have 'Ch0[V]' column")
                continue

            # Calculate mean and standard deviation
            mean_value = df['Ch0[V]'].mean()
            std_dev = df['Ch0[V]'].std()

            # Add data to our list
            data_entry = {
                'filename': filename,
                'diameter': radius,
                'angle': angle,
                'mean': mean_value,
                'std_dev': std_dev
            }

            all_data.append(data_entry)

            print(f"Processed {filename}: diameter={radius}, angle={angle}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

    if not all_data:
        print("No valid data was extracted from the files")
        return

    # Create DataFrame from extracted data
    result_df = pd.DataFrame(all_data)

    # Select and reorder columns we want for the output
    columns = ['diameter', 'angle', 'mean', 'std_dev']

    output_df = result_df[columns]

    # Create output file path
    output_path = os.path.join(folder_path, "consolidated_angle_data.csv")

    # Save to new CSV file
    output_df.to_csv(output_path, index=False)

    print(f"\nData successfully extracted and saved to {output_path}")

    return output_path


if __name__ == "__main__":
    folder_path = '..\data\Mie_angles'
    extract_data_from_csv_files(folder_path)
