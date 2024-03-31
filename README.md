## rl-final-project

# Files 
* `main.py`: Main algorithm

# To Run
Currently testing on Pong (Atari 2600 Game). 

To execute, cd into `rl-project`, install the packages from `requirements.txt`, then run `python main.py`.

# TODO 
Implement a single agent for playing multiple games. 
* Implement a Gym RL model that randomizes the game on `reset()`. [See link](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation). [Also link](https://gymnasium.farama.org/api/vector/#async-vector-env). (done) From there, we can identify how to generalize things like the Reward function and state space. 
* Implement a baseline with random action policy (each game a subset of 18 predetermined actions)
* Implement multitask agent
* Set up replay buffer


# Minor TODOS
For observation space, can preprocess to observe only the 128 Bytes of RAM of the console

# Other Resources
[Solving 57 Atari Games with a framework Agent57](https://deepmind.google/discover/blog/agent57-outperforming-the-human-atari-benchmark/)
[Solving Pong with DQN, DDPG, PPO](https://coax.readthedocs.io/en/latest/examples/atari/index.html)