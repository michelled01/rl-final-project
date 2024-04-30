## rl-final-project

Authors: Michelle Ding, Nicolas Hsu

# Introduction 
Implement a baseline and single agent for playing multiple Atari games. 

# Files 
Important folders and files
* `main.py`: Main model algorithm
* `baseline.py`: Main baseline algorithm

Other folders

* `utils`: Contains components for the Experience Replay Buffer, Neural Network, Tensorboard Data Logger, and Behavioral $\epsilon$-greedy Policy used for the Model
* `Breakout` and `Pong` folders
    * `models`: Saved network configurations for $Q$-network
    * `replay_logs`: DQN Replay Dataset storing (1/5)th of Breakout data
* `hyperparams`: baseline parameters for DQN, PPO, QRDQN PPO_LSTM, TRPO, A2C, ARS`
* `slurm-logs`: Configuration files for hyperparameter search on TACC
* `multitask_atari`: Configuration for the multitask Atari environment for preprocessing
* `logs`: Logged runs from Tensorboard setup
* `tacc_results`: Graphs generated from Tensorboard setup
* `pong-is-hard.png`: Depiction of a typical agent on A2C and ARS which both go to one side of the screen instead of moving around
* `requirements.txt`: Please install the listed packages

# To Run
To execute, cd into `rl-final-project`, install the packages from `requirements.txt`, then run `python main.py` and `baseline.py`.
To generate graphs, run `tensorboard --logdir logs`.