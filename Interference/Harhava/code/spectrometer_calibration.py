import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

mercury_filename = r'..\data\mercury_spectrum_1_close.csv'
mercury_df = pd.read_csv(mercury_filename)
mercury_df['Intensity'] = np.abs(mercury_df['Ch1[V]'])
mercury_df['Angle'] = mercury_df['Ch0[V]']

mercury_angle_data = mercury_df['Angle'].values
mercury_intensity_data = mercury_df['Intensity'].values

sodium_filename = r'..\data\Sodium_spectrum_1_close.csv'
sodium_df = pd.read_csv(sodium_filename)
sodium_df['Intensity'] = np.abs(sodium_df['Ch1[V]'])
sodium_df['Angle'] = sodium_df['Ch0[V]']

sodium_angle_data = sodium_df['Angle'].values
sodium_intensity_data = sodium_df['Intensity'].values

plt.figure(figsize=(10, 6))
plt.plot(mercury_angle_data, mercury_intensity_data, label='Mercury Data', color='blue')
plt.plot(sodium_angle_data, sodium_intensity_data, label='Sodium Data', color='orange')
plt.xlabel('Angle (degrees)')
plt.ylabel('Intensity (V)')
plt.title('Spectrometer Calibration')
plt.legend()
plt.grid()
plt.show()