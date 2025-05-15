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
    # Dictionary to map colors to wavelengths (in nm)
    color_to_wavelength = {
        'red': 640,  # Red light wavelength (approximate)
        'green': 530,  # Green light wavelength (approximate)
        'blue': 460  # Blue light wavelength (approximate)
    }

    # Dictionary to store reference measurements for each color
    reference_values = {
        'red': None,
        'green': None,
        'blue': None
    }

    reference_stds = {
        'red': None,
        'green': None,
        'blue': None
    }

    diameter_err_dict = {
        0.5: 0.05,
        0.75: 0.075,
        1: 0.1
    }

    wv_err_dict = {
        'red': 20,
        'green': 10,
        'blue': 10
    }

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
        ref_match = re.match(r'ref_(red|green|blue)', base_name, re.IGNORECASE)
        if ref_match:
            color = ref_match.group(1).lower()

            try:
                # Read the reference CSV file
                df = pd.read_csv(csv_file)

                # Check if the required column exists
                if 'Ch0[V]' not in df.columns:
                    print(f"Warning: Reference file {filename} doesn't have 'Ch0[V]' column")
                    continue

                # Calculate reference mean value
                ref_mean = df['Ch0[V]'].mean()
                ref_std = df['Ch0[V]'].std()
                reference_values[color] = ref_mean
                reference_stds[color] = ref_std

                print(f"Processed reference file for {color}: mean value = {ref_mean}")

            except Exception as e:
                print(f"Error processing reference file {filename}: {str(e)}")

    print("\nReference values:")
    for color, value in reference_values.items():
        if value is not None:
            print(f"{color}: {value}")
        else:
            print(f"{color}: Not found")

    # Now process the regular measurement files
    for csv_file in csv_files:
        # Extract filename without extension
        filename = os.path.basename(csv_file)
        base_name = os.path.splitext(filename)[0]

        # Skip reference files as we've already processed them
        if base_name.startswith('ref-'):
            continue

        # Extract radius and color using regex
        match = re.match(r'(0\.5|0\.75|1)r_(red|green|blue)', base_name)
        if not match:
            print(f"Skipping file {filename} - doesn't match expected naming pattern")
            continue

        radius = float(match.group(1))
        color = match.group(2)

        # Get wavelength for the color
        wavelength = color_to_wavelength.get(color)
        wavelength_err = wv_err_dict[color]
        radius_err = diameter_err_dict[radius]

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

            # Get reference value for this color
            ref_value = reference_values.get(color)

            # Add data to our list
            data_entry = {
                'filename': filename,
                'diameter': radius,
                'diameter_error_um2': radius_err,
                'color': color,
                'wavelength': wavelength,
                'wavelength_error_nm': wavelength_err,
                'mean': mean_value,
                'std_dev': std_dev
            }

            # Add reference value if available
            if ref_value is not None:
                data_entry['reference'] = ref_value
                data_entry['std_dev_ref'] = reference_stds[color]
                # Calculate normalized value (mean/reference)
                data_entry['normalized'] = mean_value / ref_value

            all_data.append(data_entry)

            print(f"Processed {filename}: diameter={radius}, color={color}, wavelength={wavelength}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

    if not all_data:
        print("No valid data was extracted from the files")
        return

    # Create DataFrame from extracted data
    result_df = pd.DataFrame(all_data)

    # Select and reorder columns we want for the output
    columns = ['diameter', 'diameter_error_um2', 'wavelength', 'wavelength_error_nm', 'mean', 'std_dev']
    if 'reference' in result_df.columns:
        columns.extend(['reference', 'normalized', 'std_dev_ref'])

    output_df = result_df[columns]

    # Create output file path
    output_path = os.path.join(folder_path, "consolidated_data.csv")

    # Save to new CSV file
    output_df.to_csv(output_path, index=False)

    print(f"\nData successfully extracted and saved to {output_path}")
    print(
        f"Processed {len(all_data)} regular files + {sum(1 for v in reference_values.values() if v is not None)} reference files")

    return output_path


if __name__ == "__main__":
    folder_path = '..\data\Mie'
    extract_data_from_csv_files(folder_path)
