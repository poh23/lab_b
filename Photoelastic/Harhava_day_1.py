import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\Einav\Desktop\Uni\Lab\lab_b\Photoelastic\data\Harhava_day_1.csv'
data = pd.read_csv(file_path)

# Filter relevant columns and remove rows with blank lines (zeros in the Isochromes columns)
filtered_data = data[
    (data['Num of Isochromes (on the sides)'] > 0) &
    (data['Num of Isochromes (in the center)'] > 0)
][['Name of file', 'Num of Isochromes (on the sides)', 'Num of Isochromes (in the center)', 'Compression (dL/L)']]

# Calculate the ratio for the plot
filtered_data['Isochrome Ratio'] = (
    filtered_data['Num of Isochromes (in the center)'] /
    filtered_data['Num of Isochromes (on the sides)']
)

# Separate the data into groups based on the file name
rect_clear_data = filtered_data[filtered_data['Name of file'].str.contains('Rect_clear')]
rect_1_data = filtered_data[filtered_data['Name of file'].str.contains('Rect_1')]

# Plot the data with the horizontal reference line
plt.figure(figsize=(10, 6))

# Plot the data points for Rect_1
plt.plot(rect_1_data['Compression (dL/L)'], rect_1_data['Isochrome Ratio'], 's-', label='Rect_1')

# Add a horizontal reference line at y=1 for clear_ref
plt.axhline(y=1, color='r', linestyle='--', label='Clear_ref')

# Add labels, title, and legend
plt.xlabel('Compression (dL/L)')
plt.ylabel('Isochrome Ratio (Center / Sides)')
plt.title('Isochrome Ratio vs Compression')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Re-filter the data to ensure the 'curvature of the bent structure (1/m)' column is included
filtered_data = data[
    (data['Num of Isochromes (on the sides)'] > 0) &
    (data['Num of Isochromes (in the center)'] > 0)
][['Name of file', 'Num of Isochromes (on the sides)', 'Num of Isochromes (in the center)', 
   'Compression (dL/L)', 'curvature of the bent structure (1/m)']]

# Recalculate the ratio for the plot
filtered_data['Isochrome Ratio'] = (
    filtered_data['Num of Isochromes (in the center)'] /
    filtered_data['Num of Isochromes (on the sides)']
)

# Separate the data into groups based on the file name
rect_1_data = filtered_data[filtered_data['Name of file'].str.contains('Rect_1')]

# Plot the Isochrome Ratio as a function of curvature of the bent structure
plt.figure(figsize=(10, 6))

# Plot the data points for Rect_1
plt.plot(rect_1_data['curvature of the bent structure (1/m)'], rect_1_data['Isochrome Ratio'], 's-', label='Rect_1')

# Add a horizontal reference line at y=1 for clear_ref
plt.axhline(y=1, color='r', linestyle='--', label='Clear_ref')

# Add labels, title, and legend
plt.xlabel('Curvature of the Bent Structure (1/m)')
plt.ylabel('Isochrome Ratio (Center / Sides)')
plt.title('Isochrome Ratio vs Curvature of the Bent Structure')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
