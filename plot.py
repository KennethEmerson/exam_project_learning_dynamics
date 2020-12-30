import numpy as np
import pickle
import os.path
from simulation import HunterConfig
import matplotlib.pyplot as plt

def load_hunter_config_from_bin(filename):
    """ loads a binary file (pickle) containing the hunter configuration

    Args:
        filename (string): name of the file to load

    Returns:
        [HunterConfig]: the loaded hunter configuration
    """
    with open(filename, 'rb') as hunter_config_file:
        return pickle.load(hunter_config_file)

def load_results_from_csv(filename):
    """ loads a CSV file containing the average timesteps recorded

    Args:
        filename (string): name of the file to load

    Returns:
        [numpy array]: loads the testresults as a numpy array
    """
    return np.loadtxt(filename, delimiter=';',dtype=int)

def create_data_for_one_plot(filename):
    """loads all required data to plot one lineplot

    Args:
        filename (string): file to be loaded, can be a CSV file with measurements or a Hunteconfiguration Bin file

    Returns:
        [string]: name of the policy used by the hunters
        [numpy array]: the testresults
        [int]: the total amount of training episodes used during training
    """
    if(os.path.splitext(filename)[1] ==".bin"):
        test_case = load_hunter_config_from_bin(filename)
        name = test_case.name
        total_training_episodes = test_case.total_training_episodes
        data = test_case.average_timesteps
    
    if(os.path.splitext(filename)[1] ==".csv"):
        f = open(filename)
        header = f.readline()
        name = header.split(' ')[1]
        total_training_episodes = int(header.split(' ')[-1])
        data = load_results_from_csv(filename)
    return name, data, total_training_episodes

def plot_graph(file_list):
    """[summary]

    Args:
        file_list (List):   list of all files for which a plot needs to be created, 
                            can be CSV files with measurements or a Hunteconfiguration Bin files or mixed
    """
    fig, ax = plt.subplots()

    for filename in file_list:
        name, data, total_training_episodes = create_data_for_one_plot(filename)
        print(name)
        sample_points = data.size
        episodes_x = np.linspace(0,total_training_episodes,num=sample_points)
        ax.plot(episodes_x,data,label=name,linewidth=0.6)
  
    ax.set_xlabel('number of learning episodes')
    ax.set_ylabel('Average timesteps')
    ax.legend()
    
    plt.show()


if __name__ == "__main__":
    
    # add filenames in list which you want on plot
    # both CSV results and hunterconfig binary files can be used
    file_list = [
                 #"hunters_QwPAE_20201230_000944.bin",
                 #"results_QwPAE_20201230_003849.csv",
                ]
    plot_graph(file_list)
  