
import pandas as pd
import matplotlib.pyplot as plt

#united graph
# Load the CSV file
file_path = r'C:\Users\Einav\Desktop\Uni\שנה ג\Semester_a\Lab_b1\Photoelesticity\Seperate_colors.csv'
df = pd.read_csv(file_path)
import matplotlib.pyplot as plt

# Extract unique discs and wavelengths for grouping
unique_discs = df['file name'].str.split('_').str[0].unique()
unique_wavelengths = df['Lamda (wave length nm)'].unique()

# Plot 1: N (Num of Isochromes) as a function of F/lambda for each disc
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, disc in enumerate(unique_discs):
    disc_df = df[df['file name'].str.contains(disc)]
    axes[i].scatter(disc_df['F/lambda'], disc_df['Num of Isochromes'], label=f'{disc}')
    axes[i].set_title(f'N vs F/lambda for {disc}')
    axes[i].set_xlabel('F/lambda')
    axes[i].set_ylabel('N (Num of Isochromes)')
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()

# Plot 2: N (Num of Isochromes) as a function of F/R for each wavelength
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes = axes.flatten()

for i, wavelength in enumerate(unique_wavelengths):
    wavelength_df = df[df['Lamda (wave length nm)'] == wavelength]
    axes[i].scatter(wavelength_df['F/R'], wavelength_df['Num of Isochromes'], label=f'{wavelength} nm')
    axes[i].set_title(f'N vs F/R for {wavelength} nm')
    axes[i].set_xlabel('F/R')
    axes[i].set_ylabel('N (Num of Isochromes)')
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()
