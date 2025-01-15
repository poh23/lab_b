import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\Einav\Desktop\Uni\Lab\lab_b\Photoelastic\data\Harhava_day_1.csv'
data = pd.read_csv(file_path)

# Filter relevant columns and remove rows with blank lines (zeros in the Isochromes columns)
filtered_data = data[
    (data['Num of Isochromes (on the sides)'] > 0) &
    (data['Num of Isochromes (in the center)'] > 0)
][['Name of file', 'Num of Isochromes (on the sides)', 'Num of Isochromes (in the center)',
   'Compression (dL/L)', 'Compression error', 'curvature of the bent structure (1/m)',
   'Curvature error', 'error (on the sides)', 'error (in the center)']]

# Calculate the Isochrome Ratio and its error
filtered_data['Isochrome Ratio'] = (
    filtered_data['Num of Isochromes (in the center)'] /
    filtered_data['Num of Isochromes (on the sides)']
)
filtered_data['Isochrome Error'] = (
    filtered_data['error (in the center)'] / filtered_data['Num of Isochromes (on the sides)']
) + (
    filtered_data['Num of Isochromes (in the center)'] *
    filtered_data['error (on the sides)'] /
    filtered_data['Num of Isochromes (on the sides)']**2
)

# Separate the data into groups based on the file name
rect_clear_data = filtered_data[filtered_data['Name of file'].str.contains('Rect_clear')]
rect_1_data = filtered_data[filtered_data['Name of file'].str.contains('Rect_1')]

# Plot the Isochrome Ratio as a function of Compression with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(
    rect_1_data['Compression (dL/L)'],
    rect_1_data['Isochrome Ratio'],
    yerr=rect_1_data['Isochrome Error'],
    xerr=rect_1_data['Compression error'],
    fmt='s-', label='Rect_1', capsize=5
)
plt.axhline(y=1, color='r', linestyle='--', label='Clear_ref')
plt.xlabel('Compression (dL/L)')
plt.ylabel('Isochrome Ratio (Center / Sides)')
plt.title('Isochrome Ratio vs Compression with Errors')
plt.legend()
plt.grid(True)
plt.show()

# Plot the Isochrome Ratio as a function of curvature with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(
    rect_1_data['curvature of the bent structure (1/m)'],
    rect_1_data['Isochrome Ratio'],
    yerr=rect_1_data['Isochrome Error'],
    xerr=rect_1_data['Curvature error'],
    fmt='s-', label='Rect_1', capsize=5
)
plt.axhline(y=1, color='r', linestyle='--', label='Clear_ref')
plt.xlabel('Curvature of the Bent Structure (1/m)')
plt.ylabel('Isochrome Ratio (Center / Sides)')
plt.title('Isochrome Ratio vs Curvature of the Bent Structure with Errors')
plt.legend()
plt.grid(True)
plt.show()
