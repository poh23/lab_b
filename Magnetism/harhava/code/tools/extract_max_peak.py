from .load_csv_to_dataframe import load_csv_to_dataframe
# Function to extract metadata and data from a CSV file
def extract_max_peak(file_path):
    data = load_csv_to_dataframe(file_path)

    # Splitting the original data into two sections for each channel
    channel_1 = data.iloc[:, :5]  # First 6 columns for Channel 1
    channel_2 = data.iloc[:, 6:11]  # Last 6 columns for Channel 2

    # Renaming columns for better readability
    channel_1.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']
    channel_2.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']

    channel_1 = channel_1[['T', 'V']]
    channel_2 = channel_2[['T', 'V']]

    # calculating average peak voltage for each channel
    ch1_max_peak_voltage = channel_1['V'].max()
    ch2_max_peak_voltage = channel_2['V'].max()

    return ch1_max_peak_voltage, ch2_max_peak_voltage