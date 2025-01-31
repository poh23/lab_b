import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


colors = {460: 'b', 525: 'g', 625: 'r'}
radii = {"disc1": 3.75, "disc2": 6.25, "disc3": 5}
font = {'family': 'serif', 'size': 16}
x_errors ={"disc1": 75, "disc2": 165, "disc3": 165, "disc4": 1.5}
lambda_fits = {'disc1': [1.4264179 , 5.99410476], 'disc2': [1.77477104, 6.99310963], 'disc3': [2.24651978, 4.27873176], 'disc4': [38.262, 4.086]}
R_fits = {460: [0.00928439, 8.38719338], 525: [0.00797887, 6.22926252], 625: [0.00871047, 7.37956197]}

def extract_data():
    # Load the CSV file
    file_path = r'data/number_of_isochromes.csv'
    df = pd.read_csv(file_path)

    # Extract unique discs
    unique_discs = df['file name'].str.split('_').str[0].unique()
    unique_wavelengths = df['Lamda (wave length nm)'].unique()

    # Separate 'disc4' for individual plotting
    disc4_df = df[df['file name'].str.contains('disc4')]
    other_discs = [disc for disc in unique_discs if disc != 'disc4']

    return df, unique_discs, unique_wavelengths, disc4_df, other_discs


# plot 1 f/lamda plots for discs one through three
def plot_1(ax):
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, disc in enumerate(other_discs):
        disc_df = df[df['file name'].str.contains(disc)]
        for wavelength in unique_wavelengths:
            wavelength_df = disc_df[disc_df['Lamda (wave length nm)'] == wavelength]
            axes[i].errorbar(wavelength_df['F/lambda'], wavelength_df['Num of Isochromes'], yerr=wavelength_df['error'], xerr= x_errors[disc]/wavelength, fmt='o', label=f'{wavelength} nm', markersize=5,
                             capsize=1, elinewidth=1, color=colors[wavelength])
            # Add linear fit
        fit = np.polyfit(disc_df['F/lambda'], disc_df['Num of Isochromes'], 1, cov=True)
        p = np.poly1d(fit[0])
        intercept_var, slope_var = np.sqrt(np.diag(fit[1]))
        axes[i].plot(disc_df['F/lambda'], p(disc_df['F/lambda']), "--", linewidth=0.9, color='black',
                     label=f'Linear fit: {p[1]:.2f}x + {p[0]:.2f}')
        print(f'{disc} linear fit: {p[1]:.5f}x + {p[0]:.5f}')

        # Calculate R squared
        values = disc_df['Num of Isochromes'].values
        residuals = values - p(disc_df['F/lambda'])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f'Slope error: {slope_var:.4f}')
        print(f'Intercept error: {intercept_var:.4f}')
        print(f'R squared: {r_squared:.4f}')

        fits = {disc: fit[0]}
        print(fits)

        axes[i].set_title(f'R = {radii[disc]} cm', fontdict=font)
        axes[i].set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
        axes[i].set_ylabel('N', fontdict=font)
        axes[i].legend(prop={'family': 'serif', 'size': 10})
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].annotate(f"{chr(65 + i)}",  # This gives 'a', 'b', 'c'...
                    xy=(-0.1, 0.98),  # Relative position (2% from left, 98% from bottom)
                    xycoords='axes fraction',
                    font=font,
                    fontsize=16,
                    fontweight='bold',
                    ha='left',
                    va='top')

    plt.tight_layout()
    plt.savefig('graphs/f_lambda_plots_discs1-3.png')

# plot 2 f/r plots for discs one through three
def plot_2():
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()
    all_disc_df = df[df['file name'].str.contains('|'.join(other_discs))]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, wavelength in enumerate(unique_wavelengths):
        wavelength_df = all_disc_df[all_disc_df['Lamda (wave length nm)'] == wavelength]
        for disc in other_discs:
            disc_df = wavelength_df[wavelength_df['file name'].str.contains(disc)]
            axes[i].errorbar(disc_df['F/R'], disc_df['Num of Isochromes'], yerr=disc_df['error'], xerr= x_errors[disc]/radii[disc], fmt='o', label=f'{radii[disc]} cm', markersize=5,
                             capsize=1, elinewidth=1)
        # Add linear fit
        fit = np.polyfit(wavelength_df['F/R'], wavelength_df['Num of Isochromes'], 1, cov=True)
        p = np.poly1d(fit[0])
        intercept_var, slope_var = np.sqrt(np.diag(fit[1]))
        axes[i].plot(wavelength_df['F/R'], p(wavelength_df['F/R']), "--", linewidth=0.9, color='black',
                     label=f'Linear fit: {p[1]:.2f}x + {p[0]:.2f}')
        print(fr'$\lambda$ = {wavelength} linear fit: {p[1]:.5f}x + {p[0]:.5f}')

        # Calculate R squared
        values = wavelength_df['Num of Isochromes'].values
        residuals = values - p(wavelength_df['F/R'])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f'Slope error: {slope_var:.4f}')
        print(f'Intercept error: {intercept_var:.4f}')
        print(f'R squared: {r_squared:.4f}')

        fits = {wavelength: fit[0]}
        print(fits)

        axes[i].set_title(fr'$\lambda$={wavelength} nm', fontdict=font)
        axes[i].set_xlabel(r'$\frac{F}{R}$ ($\frac{N}{cm}$)', fontdict=font)
        axes[i].set_ylabel('N', fontdict=font)
        axes[i].legend(prop={'family': 'serif', 'size': 10})
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].annotate(f"{chr(65 + i)}",  # This gives 'a', 'b', 'c'...
                         xy=(-0.1, 0.98),  # Relative position (2% from left, 98% from bottom)
                         xycoords='axes fraction',
                         font=font,
                         fontsize=16,
                         fontweight='bold',
                         ha='left',
                         va='top')

    plt.tight_layout()
    plt.savefig('graphs/f_R_plots_discs1-3.png')

# # Plot 3: Separate plots for disc4
def plot_3():
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()

    fig, axes = plt.subplots(1, 1, figsize=(4, 5), sharex=True, sharey=True)

    for wavelength in unique_wavelengths:
        wavelength_df = disc4_df[disc4_df['Lamda (wave length nm)'] == wavelength]
        axes.errorbar(wavelength_df['F/lambda'], wavelength_df['Num of Isochromes'], yerr=wavelength_df['error'], xerr=x_errors['disc4']/wavelength ,fmt='o', label=f'{wavelength} nm', markersize=5,
                         capsize=1, elinewidth=1, color=colors[wavelength])
        # Add linear fit
    fit = np.polyfit(disc4_df['F/lambda'], disc4_df['Num of Isochromes'], 1, cov=True)
    p = np.poly1d(fit[0])
    intercept_var, slope_var = np.sqrt(np.diag(fit[1]))
    axes.plot(disc4_df['F/lambda'], p(disc4_df['F/lambda']), "--", linewidth=0.9, color='black',
                 label=f'Linear fit: {p[1]:.2f}x + {p[0]:.2f}')
    print(f'disc4 linear fit: {p[1]:.5f}x + {p[0]:.5f}')

    # Calculate R squared
    values = disc4_df['Num of Isochromes'].values
    residuals = values - p(disc4_df['F/lambda'])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f'Slope error: {slope_var:.4f}')
    print(f'Intercept error: {intercept_var:.4f}')
    print(f'R squared: {r_squared:.4f}')

    fits = {'disc4': fit[0]}
    print(fits)

    axes.set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
    axes.set_ylabel('N', fontdict=font)
    axes.legend(prop={'family': 'serif', 'size': 10})
    axes.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig('graphs/disc4_plots.png')


# Plot 4: white discs
def plot_4():
    font = {'family': 'serif', 'size': 18}
    colors = ['blue', 'green', 'red']
    wave_lengths = {'blue': 460, 'green': 525, 'red': 625}
    # Load the CSV file
    file_path = r'data/white_isochromes_new.csv'
    df = pd.read_csv(file_path)

    unique_discs = df['white files'].str.split('_').str[0].unique()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, disc in enumerate(unique_discs):
        disc_df = df[df['white files'].str.contains(disc)]
        for color in colors:
            axes[i].errorbar(disc_df['F']/wave_lengths[color], disc_df[f'N {color} isochromes'], yerr=disc_df[f'{color}_error'], xerr=x_errors[disc]/wave_lengths[color], fmt='o', label=f'{wave_lengths[color]} nm', markersize=5,
                             capsize=1, elinewidth=1, color=color)
        # Add linear fit
        p = np.poly1d(lambda_fits[disc])
        axes[i].plot(disc_df['F']/wave_lengths['blue'], p(disc_df['F']/wave_lengths['blue']), "--", linewidth=0.9, color='black')

        axes[i].set_title(f'Disc {disc[-1]}', fontdict=font)
        axes[i].set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
        axes[i].set_ylabel('N', fontdict=font)
        axes[i].legend(prop={'family': 'serif', 'size': 14})
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].annotate(f"{chr(65 + i)}",  # This gives 'a', 'b', 'c'...
                         xy=(-0.1, 0.98),  # Relative position (2% from left, 98% from bottom)
                         xycoords='axes fraction',
                         font=font,
                         fontsize=16,
                         fontweight='bold',
                         ha='left',
                         va='top')

    plt.tight_layout(h_pad=4.0, w_pad=4.0)
    plt.savefig('graphs/white_f_lambda_plots.png')

def plot_5():
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()

    fig1, axes1 = plt.subplots(1, 1, figsize=(6, 5))
    fig2, axes2 = plt.subplots(1, 1, figsize=(6, 5))

    discs3_df = df[df['file name'].str.contains('|'.join(other_discs))]

    for disc in other_discs:
        disc_df = df[df['file name'].str.contains(disc)]
        p = np.poly1d(lambda_fits[disc])
        axes1.plot(disc_df['F/lambda'], p(disc_df['F/lambda']), linewidth=0.9,
                     label=f'R = {radii[disc]} cm')

    for wavelength in unique_wavelengths:
        wavelength_df = discs3_df[discs3_df['Lamda (wave length nm)'] == wavelength]
        p = np.poly1d(R_fits[wavelength])
        axes2.plot(wavelength_df['F/R'], p(wavelength_df['F/R']), linewidth=0.9,
                     label=fr'$\lambda$ = {wavelength} nm', color = colors[wavelength])


    axes1.set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
    axes1.set_ylabel('N', fontdict=font)
    axes2.set_xlabel(r'$\frac{F}{R}$ ($\frac{N}{cm}$)', fontdict=font)
    axes2.set_ylabel('N', fontdict=font)

    for idx, ax in enumerate([axes1, axes2]):
        ax.legend(prop = {'family': 'serif', 'size': 12})
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f'graphs/fit_plot_1.png')
    fig2.savefig(f'graphs/fit_plot_2.png')







