from pathlib import Path
import pandas as pd

# Define the file path using Path and raw string to avoid unicode errors
file_path = Path(r"..\data\wideslit_vertical_aper_020425_2_-8vmax.txt")

# Read the file using whitespace as separator (new recommended method)
df = pd.read_csv(file_path, sep=r'\s+')

# Create the CSV path
csv_path = file_path.with_suffix(".csv")

# Save as CSV
df.to_csv(csv_path, index=False)

print(f"CSV saved to: {csv_path}")
