import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\Einav\Desktop\Uni\שנה ג\Semester_a\Lab_b1\Photoelesticity\Seperate_colors.csv'
df = pd.read_csv(file_path)

# Extract unique discs
unique_discs = df['file name'].str.split('_').str[0].unique()

# Separate 'disc4' for individual plotting
disc4_df = df[df['file name'].str.contains('disc4')]
other_discs = [disc for disc in unique_discs if disc != 'disc4']

# Plot 1: Combined plot for other discs
fig, axes = plt.subplots(1, len(other_discs), figsize=(15, 5), sharex=True, sharey=True)
axes = axes.flatten()

for i, disc in enumerate(other_discs):
    disc_df = df[df['file name'].str.contains(disc)]
    axes[i].scatter(disc_df['F/lambda'], disc_df['Num of Isochromes'], label=f'{disc}')
    axes[i].set_title(f'N vs F/lambda for {disc}')
    axes[i].set_xlabel('F/lambda')
    axes[i].set_ylabel('N (Num of Isochromes)')
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()

# Plot 2: Separate plot for disc4
plt.figure(figsize=(6, 5))
plt.scatter(disc4_df['F/lambda'], disc4_df['Num of Isochromes'], label='disc4', color='red')
plt.title('N vs F/lambda for disc4')
plt.xlabel('F/lambda')
plt.ylabel('N (Num of Isochromes)')
plt.legend()
plt.grid()
plt.show()

# Exclude 'disc4' from the F/R plots
filtered_df = df[~df['file name'].str.contains('disc4')]

# Plot 3: N (Num of Isochromes) as a function of F/R for each wavelength
unique_wavelengths = filtered_df['Lamda (wave length nm)'].unique()
fig, axes = plt.subplots(1, len(unique_wavelengths), figsize=(15, 5), sharex=True, sharey=True)
axes = axes.flatten()

for i, wavelength in enumerate(unique_wavelengths):
    wavelength_df = filtered_df[filtered_df['Lamda (wave length nm)'] == wavelength]
    axes[i].scatter(wavelength_df['F/R'], wavelength_df['Num of Isochromes'], label=f'{wavelength} nm')
    axes[i].set_title(f'N vs F/R for {wavelength} nm')
    axes[i].set_xlabel('F/R')
    axes[i].set_ylabel('N (Num of Isochromes)')
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()


