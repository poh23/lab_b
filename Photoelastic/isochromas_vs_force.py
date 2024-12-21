import pandas as pd
import matplotlib.pyplot as plt

#united graph
# Load the CSV file
file_path = r'C:\Users\Einav\Desktop\Uni\שנה ג\Semester_a\Lab_b1\Photoelesticity\Seperate_colors.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())


# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot N vs F/R
ax1.errorbar(df['F/R'], 
            df['Num of Isochromes'],
            yerr=abs(df['error']),
            fmt='bo',
            capsize=5,
            label='Data points')

ax1.set_xlabel('F/R (N/cm)', fontsize=12)
ax1.set_ylabel('Number of Isochromes (N)', fontsize=12)
ax1.set_title('N vs F/R', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Plot N vs F/lambda
ax2.errorbar(df['F/lambda'], 
            df['Num of Isochromes'],
            yerr=abs(df['error']),
            fmt='ro',
            capsize=5,
            label='Data points')

ax2.set_xlabel('F/λ (N/nm)', fontsize=12)
ax2.set_ylabel('Number of Isochromes (N)', fontsize=12)
ax2.set_title('N vs F/λ', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# Add minor ticks for better precision
ax1.minorticks_on()
ax2.minorticks_on()

# Add a main title
fig.suptitle('Relationship between Number of Isochromes and Force Parameters', 
             fontsize=16, y=1.05)

# Adjust layout
plt.tight_layout()
plt.show()






#option 2: eperated graphs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with subplots (2x4 grid)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Relationship between Number of Isochromes and Force Parameters by Disc', fontsize=16)

# Get unique disc numbers
disc_numbers = [1, 2, 3, 4]

# Plots for F/R will be on top row, F/lambda on bottom row
for i, disc_num in enumerate(disc_numbers):
    # Filter data for current disc
    disc_data = df[df['file name'].str.contains(f'disc{disc_num}')]
    
    # Plot N vs F/R (top row)
    axes[0, i].errorbar(disc_data['F/R'], 
                       disc_data['Num of Isochromes'],
                       yerr=abs(disc_data['error']),
                       fmt='bo',
                       capsize=5,
                       label='Data points')
    
    axes[0, i].set_xlabel('F/R (N/cm)', fontsize=10)
    axes[0, i].set_ylabel('Number of Isochromes (N)', fontsize=10)
    axes[0, i].set_title(f'Disc {disc_num}: N vs F/R', fontsize=12)
    axes[0, i].grid(True, linestyle='--', alpha=0.7)
    axes[0, i].legend()
    axes[0, i].minorticks_on()
    
    # Plot N vs F/lambda (bottom row)
    axes[1, i].errorbar(disc_data['F/lambda'], 
                       disc_data['Num of Isochromes'],
                       yerr=abs(disc_data['error']),
                       fmt='ro',
                       capsize=5,
                       label='Data points')
    
    axes[1, i].set_xlabel('F/λ (N/nm)', fontsize=10)
    axes[1, i].set_ylabel('Number of Isochromes (N)', fontsize=10)
    axes[1, i].set_title(f'Disc {disc_num}: N vs F/λ', fontsize=12)
    axes[1, i].grid(True, linestyle='--', alpha=0.7)
    axes[1, i].legend()
    axes[1, i].minorticks_on()

# Adjust layout
plt.tight_layout()
plt.show()
