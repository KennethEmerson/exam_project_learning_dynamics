import numpy as np
import pickle
import math
import os.path
import matplotlib.pyplot as plt
from simulation import HunterConfig, Centralized_Config

def load_hunter_config_from_bin(filename: str):
    """
    Load a pickle binary file containing the hunter configuration.

    :param filename: The name of the file to load

    :return: The loaded hunter configuration.
    """
    with open(filename, 'rb') as hunter_config_file:
        return pickle.load(hunter_config_file)


def load_results_from_csv(filename: str):
    """
    Load a CSV file containing the average time steps recorded.

    :param filename: Name of the file to load.

    :return: A numpy array with the average time steps.
    """
    return np.loadtxt(filename, delimiter=';', dtype=int)


def create_data_for_one_plot(filename: str) -> (str, np.array, int):
    """
    Loads all the required data to plot one lineplot.

    :param filename: The file to be loaded, can be a CSV file with
        measurements or a Hunteconfiguration bin file.

    :return: A triple with (in that order): the name of the policy
        used by the hunters, the test results, the total amount of
        training episodes used during training.
    """
    if os.path.splitext(filename)[1] == ".bin":
        test_case = load_hunter_config_from_bin(filename)
        name = test_case.name
        total_training_episodes = test_case.total_training_episodes
        average_data = test_case.average_time_steps
        if hasattr(test_case,'std_time_steps'):
            std_data = test_case.std_time_steps
            max_data = test_case.max_time_steps
            min_data = test_case.min_time_steps
            mae_data = test_case.mae_time_Steps
        else: 
            std_data = None
            max_data = None
            min_data = None
            mae_data = None

    if os.path.splitext(filename)[1] == ".csv":
        f = open(filename)
        header = f.readline()
        name = header.split(' ')[1]
        total_training_episodes = int(header.split(' ')[-1])
        average_data = load_results_from_csv(filename)
        std_data = None
        max_data = None
        min_data = None
        mae_data = None
    return name, average_data, std_data, max_data, min_data, mae_data, total_training_episodes


def plot_graph(file_list: [str]):
    """[summary]
    Plot the graphs in the file list.

    :param file_list: List of all files for which a plot needs to be
        created. They can be CSV files with measurements, a Hunteconfiguration
        bin files or a mix of both.
    """
    color_list=['b','g','r','c','m']
    fig, ax = plt.subplots()
    max_x_values = np.zeros(len(file_list))
    max_y_values = np.zeros(len(file_list))

    for file_index, filename in enumerate(file_list):
        name, average_data, std_data, max_data, min_data, mae_data,total_training_episodes = create_data_for_one_plot(filename)
        max_x_values[file_index] = total_training_episodes
        max_y_values[file_index] = int(math.ceil(np.max(average_data) / 100)) * 100
        sample_points = average_data.size
        episodes_x = np.linspace(0, total_training_episodes, num=sample_points)
        ax.plot(episodes_x, average_data, label=name, linewidth=0.6,color=color_list[file_index])
        if std_data is not None:
            ax.fill_between(episodes_x, average_data+std_data, average_data-std_data, color=color_list[file_index], alpha=0.4)
            #ax.plot(episodes_x, max_data,linestyle=':', linewidth=0.6,color=color_list[file_index])
            #ax.plot(episodes_x, min_data,linestyle=':', linewidth=0.6,color=color_list[file_index])
            #ax.fill_between(episodes_x, max_data, min_data, color=color_list[file_index], alpha=0.3)
    ax.set_xlabel('number of learning episodes')
    ax.set_ylabel('Average time steps')
    ax.legend()

    plt.xticks(np.arange(0, 2500, 500))
    plt.yticks(np.arange(0, np.max(max_y_values), 100))
    plt.xlim(0, 2000)
    plt.ylim(0, np.max(max_y_values))
    plt.show()


if __name__ == "__main__":
    # add filenames in list which you want on plot
    # both CSV results and hunterconfig binary files can be used
    file_list = [
        "hunters_Q-learning with randomly action estimation_02012021_2200.bin",
        "hunters_Q-learning with proposed action estimation_02012021_2041.bin",
        "hunters_Centralized Q-learning_02012021_2115.bin",
    ]
    plot_graph(file_list)
