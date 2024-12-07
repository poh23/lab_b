import matplotlib.pyplot as plt
from tools.extract_data_from_files import extract_data_from_files
import matplotlib
matplotlib.use('TkAgg')

def show_graphs_of_H_by_time():
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    axes1 = axes1.ravel()

    file_path_4_space = f"../data/high/Harhava_material2_1sheet_0_space_high.csv"
    channel_1_4_space, channel_2_4_space = extract_data_from_files(file_path_4_space)

    file_path_6_space = f"../data/high/Harhava_material2_1sheet_standing.csv"
    channel_1_6_space, channel_2_6_space = extract_data_from_files(file_path_6_space)

    axes1[0].plot(channel_1_4_space['T'], channel_1_4_space['V'], label='H 4 space', c='g')
    axes1[0].plot(channel_1_6_space['T'], channel_1_6_space['V'], label='H 6 space', c='orange', linestyle='--')
    axes1[1].plot(channel_2_4_space['T'], channel_2_4_space['V'], label='B 4 space', c='g')
    axes1[1].plot(channel_2_6_space['T'], channel_2_6_space['V'], label='B 6 space', c='orange', linestyle='--')

    for ax in axes1:
        ax.legend()

    plt.show()

show_graphs_of_H_by_time()