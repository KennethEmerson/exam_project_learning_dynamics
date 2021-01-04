# Multi-Agent Reinforcement Learning: An Approach Based on the Other Agent's Internal Model

## Description

This repository contains all the that was code used to analyze and try to replicate the results of Nagayuki, Y., Ishii, S., and Doya, K. paper, "Multi-agent reinforcement
learning: an approach based on the other agent's internal model.", 2000, In Proceedings Fourth International Conference on MultiAgent Systems, pages 215–221.

This review is done in the context of the Learning Dynamics course, given jointly by the Université Libre de Bruxelles and Vrije Universiteit Brussel in Belgium.

## Installation

To install this project, you will first need to install Python 3. This can be done on python [official website](https://www.python.org/downloads/). 

Once Python 3 is installed, you will need to install the [matplotlib](https://matplotlib.org/) and [numpy](https://numpy.org/) librairies.

You can install matplotlib by using the following command on a terminal:

```sh
pip install matplotlib
```

and numpy as follow:

```sh
pip install numpy
```

## Usage

### Running a simulation
You can directly run a simulation by launching the `main.py` file. This can be done with the following command line:

```sh
python main.py
```

If you want to reproduce our results, you will need to use the corresponding simulation contained within the simulation/ directory. This can be done by starting a terminal on the main directory and then start the simulation that interest you (let's say simulation_figure_X.py) in the following way:

```sh
python -m simulation.simulation_figure_X
```

The simulation figures numbers correspond to the ones in the original paper of Nagayuki et al. (2000). The numbers correspond to:

- 5 : Centralized Q-learning (CQ) method vs multi-agent Q-learning method with proposed action estimation (QwPAE) and random action estimation (QwRAE);
- 6 : QwPAE with two hunters having different reward functions;
- 7 : QwPAE vs Multi-agent Q-learning method with self-model based action estimation (QwSAE) on a homogeneous game;
- 8 : QwPAE vs QwSAE on a game with different goals.

Notice that our program only generate one agent at a time so you will have to uncomment one of them at the time to generate the results. Those results are stored into a pickle .bin file and a .csv file that can in turn be plotted by using `plot.py` (you have to replace in the code the files that you want to use).

### Visual game episode

You can see an animation of the agents of your choice, hunting a prey, by launching the `animation.py` file.




