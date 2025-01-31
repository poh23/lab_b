import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import curve_fit

def linear_fit_no_intercept(x, m):
    return m * x - 0.05

matplotlib.use('TkAgg')

# Rose Pine Moon Theme Colors
rose_pine_colors = {
    "background": "#232136",  # Base background
    "grid": "#2A273F",        # Overlay/grid
    "muted": "#393552",       # Muted text/lines
    "rose": "#EB6F92",        # Accent color
    "gold": "#F6C177",        # Accent color
    "pine": "#31748F",        # Accent color
    "foam": "#9CCFD8",        # Accent color
    "iris": "#C4A7E7",        # Accent color
    "love": "#E46876",        # Accent color
}

font = {'family': 'serif', 'size': 16}
bar_cross_section = 1  # 1 cm^2

# Set global dark theme
plt.rcParams.update({
    "figure.facecolor": rose_pine_colors["background"],
    "axes.facecolor": rose_pine_colors["background"],
    "axes.edgecolor": rose_pine_colors["muted"],
    "axes.labelcolor": rose_pine_colors["foam"],
    "xtick.color": rose_pine_colors["foam"],
    "ytick.color": rose_pine_colors["foam"],
    "grid.color": rose_pine_colors["grid"],
    "text.color": rose_pine_colors["foam"],
    "legend.frameon": False,
    "legend.fontsize": 12,
    "font.family": "serif",
})

def extract_data():
    # Load the CSV file
    file_path = r'data/Harhava Data Pt. 2 - Sheet1 (5).csv'
    df = pd.read_csv(file_path)

    column_name = 'Name of file'

    # Extract material (e.g., carb_1, carb_2)
    df['Material'] = df[column_name].str.extract(r'^(carb_\d+)')

    # Extract measurement suffix (_1, _2, _3, or nothing)
    df['Measurement'] = df[column_name].str.extract(r'_(\d+)$')

    # Replace NaN in the Measurement column with '0' to indicate no suffix
    df['Measurement'] = df['Measurement'].fillna('0')

    return df


def plot_strain_isochromatics_graph():
    df = extract_data()

    lengths = {'carb_1': '100 mm', 'carb_2': '140 mm'}

    fig, ax = plt.subplots()
    for i, material in enumerate(df['Material'].unique()):
        df_subset = df[df['Material'] == material]
        df_subset = df_subset[df_subset['Measurement'] == '0']
        ax.errorbar(
            df_subset['Compression  (dL/L)'] * 100,
            df_subset['Num of Isochromes (center cross - section )'],
            fmt='o',
            xerr=df_subset['Compression error'] * 100,
            yerr=df_subset['error (center cross-section)'],
            label=lengths[material],
            markersize=5,
            capsize=3,
            elinewidth=1.5,
            color=rose_pine_colors["love"] if i == 0 else rose_pine_colors["gold"]
        )

    # Configure plot
    ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    ax.axvline(x=1.0, color=rose_pine_colors["gold"], linestyle='--', linewidth=1.2)
    ax.axvline(x=1.2925, linestyle='--', linewidth=1.2, color=rose_pine_colors["love"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.savefig('harhava-graphs/strain_isochromatics_rose_pine.png', dpi=300)


def plot_ext_stress_vs_strain():
    df = extract_data()

    lengths = {'carb_1': '100 mm', 'carb_2': '140 mm'}

    fig, ax = plt.subplots()
    for i, material in enumerate(df['Material'].unique()):
        df_subset = df[df['Material'] == material]
        df_subset = df_subset[df_subset['Measurement'] == '0']
        ax.errorbar(
            df_subset['Compression  (dL/L)'] * 100,
            df_subset['Force (N)'] / (100 * bar_cross_section),
            fmt='o',
            xerr=df_subset['Compression error'] * 100,
            label=lengths[material],
            markersize=5,
            capsize=3,
            elinewidth=1.5,
            color=rose_pine_colors["love"] if i == 0 else rose_pine_colors["gold"]
        )

    # Configure plot
    ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
    ax.axvline(x=1.0, color=rose_pine_colors["gold"], linestyle='--', linewidth=1.2)
    ax.axvline(x=1.2925, linestyle='--', linewidth=1.2, color=rose_pine_colors["love"])
    ax.set_ylabel(r'$\sigma_{ext}$ $[MPa]$', fontdict=font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.savefig('harhava-graphs/stress_strain_rose_pine.png', dpi=300)

def plot_isochromes():
    # Load the data
    file_path = 'data/iso_vs_y.csv'
    data = pd.read_csv(file_path)

    # Get unique values of F
    unique_comp_values = [0.17, 0.25, 0.33]

    # Initialize the plot
    plt.figure()

    colors = ["pine", "foam", "iris", "love"]

    # Plot data for each value of F with error bars
    for i, F_value in enumerate(unique_comp_values):
        subset = data[data['compression'] == F_value]
        plt.errorbar(
            subset['distance (cm)'],
            subset['num isochromes until distance'],
            xerr=subset['dist error'],
            yerr=subset['error'],
            label=f"{F_value} %",
            fmt='o',
            color=rose_pine_colors[colors[i]],
            markersize=5,
            capsize=3,
            elinewidth=1.5
        )
        fit = np.polyfit(subset['distance (cm)'], subset['num isochromes until distance'], 1, cov=True, w=np.ones_like(subset['distance (cm)']))
        p = np.poly1d(fit[0])
        intercept_var, slope_var = np.sqrt(np.diag(fit[1]))

        plt.plot(
            np.sort(subset['distance (cm)']),
            p(np.sort(subset['distance (cm)'])),
            "--",
            linewidth=1.5,
            color=rose_pine_colors[colors[i]]
        )

        # Calculate R squared
        residuals = subset['num isochromes until distance'] - p(subset['distance (cm)'])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((subset['num isochromes until distance'] - np.mean(subset['num isochromes until distance'])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # # Perform linear fit without intercept
        # popt, pcov = curve_fit(linear_fit_no_intercept, subset['distance (cm)'],subset['num isochromes until distance'])
        # m = popt[0]
        # m_cov = np.sqrt(np.diag(pcov))[0]  # Extract the covariance of the slope
        #
        # # Calculate R-squared
        # residuals = subset['num isochromes until distance'] - linear_fit_no_intercept(subset['distance (cm)'], m)
        # ss_res = np.sum(residuals ** 2)
        # ss_tot = np.sum(
        #     (subset['num isochromes until distance'] - np.mean(subset['num isochromes until distance'])) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        #
        # # Generate fit line
        # x_fit = np.linspace(subset['distance (cm)'].min(), subset['distance (cm)'].max(), 100)
        # y_fit = linear_fit_no_intercept(x_fit, m)
        #
        # # Plot the fit line
        # plt.plot(x_fit, y_fit, linestyle='--', color=rose_pine_colors[colors[i]],
        #         label=f"Fit F = {F_value} (m={m:.2f} ± {m_cov:.2f}, R²={r_squared:.2f})")

        print(f'Compression: {F_value}')
        print(f'Global linear fit: {p[0]:.5f}x + {p[1]:.5f}')
        print(f'Slope error: {slope_var:.4f}')
        print(f'Intercept error: {intercept_var:.4f}')
        print(f'R squared: {r_squared:.4f}')

    # Add labels, legend, and title
    plt.xlabel('y [cm]', fontsize=12, fontweight='bold')
    plt.ylabel('N', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(title=r'$\varepsilon$ ', fontsize=10, title_fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.savefig('harhava-graphs/y_iso_rose_pine.png', dpi=300, facecolor=rose_pine_colors["background"])


def plot_ext_stress_vs_strain_w_derivative():
    df = extract_data()

    fig, ax = plt.subplots()
    df_carb2 = df[(df['Material'] == 'carb_2') & (df['Measurement'] == '0')]
    x = df_carb2['Compression  (dL/L)'] * 100
    y = df_carb2['Force (N)'] / (100 * bar_cross_section)

    # Calculate numerical derivative
    dydx = np.gradient(y, x)

    ax.scatter(x, dydx, color=rose_pine_colors["love"], s=20, label=r"($\frac{d\sigma_{ext}}{d\varepsilon}$) of 140 mm")
    ax.axvline(x=1.0, color=rose_pine_colors["love"], linestyle='--', linewidth=1.2)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
    ax.set_ylabel(r'$\frac{d\sigma_{ext}}{d\varepsilon}$', fontdict=font)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('harhava-graphs/stress_strain_derivative_rose_pine.png', dpi=300)

def plot_delta_N_vs_strain():
    df = extract_data()

    material = 'carb_2'
    for measurement in [0, 2, 3]:
        df_subset = df[(df['Material'] == material)]
        df_subset = df_subset[df_subset['Measurement'] == str(measurement)]

        # Calculate Delta N left and right with errors
        df_subset['Delta N left'] = df_subset['Num of Isochromes (center cross - section )'] - df_subset['Num of Isochromes left side (vertically)']
        df_subset['Delta N left error'] = np.sqrt(df_subset['error (center cross-section)']**2 + df_subset['error left (vertically)']**2)
        df_subset['Delta N right'] = df_subset['Num of Isochromes (center cross - section )'] - df_subset['Num of Isochromes right side (vertically)']
        df_subset['Delta N right error'] = np.sqrt(df_subset['error (center cross-section)']**2 + df_subset['error right (vertically)']**2)

        # Create a figure for each measurement
        fig, ax = plt.subplots()

        # Plot Delta N left
        ax.errorbar(
            df_subset['Compression  (dL/L)'] * 100,
            df_subset['Delta N left'],
            fmt='o',
            xerr=df_subset['Compression error'] * 100,
            yerr=df_subset['Delta N left error'],
            label=r'$\Delta N_{left}$',
            markersize=5,
            capsize=3,
            elinewidth=1.5,
            color=rose_pine_colors["pine"]
        )

        # Plot Delta N right
        ax.errorbar(
            df_subset['Compression  (dL/L)'] * 100,
            df_subset['Delta N right'],
            fmt='s',
            xerr=df_subset['Compression error'] * 100,
            yerr=df_subset['Delta N right error'],
            label=r'$\Delta N_{right}$',
            markersize=5,
            capsize=3,
            elinewidth=1.5,
            color=rose_pine_colors["foam"]
        )

        # Configure the plot
        ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
        ax.set_ylabel(r'$\Delta N$', fontdict=font)
        ax.set_xlim(-0.05, 1.6)
        ax.set_ylim(-4, 40)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc="upper left", fontsize=12)

        # Save the figure with dark background
        plt.tight_layout()
        plt.savefig(f'harhava-graphs/strain_delta-isochromatics-mes-{measurement}_rose_pine.png', dpi=300, facecolor=rose_pine_colors["background"])



# Uncomment to test individual plots

#plot_strain_isochromatics_graph()
#plot_ext_stress_vs_strain()
#plot_ext_stress_vs_strain_w_derivative()
#plot_delta_N_vs_strain()

plot_isochromes()
