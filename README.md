## rl-final-project

# Files 
* `dqn_atari.py`: Main algorithm

# To Run
Currently testing on Pong (Atari 2600 Game). 

To execute, cd into `rl-final-project` and run
```
python dqn_atari.py --env-id PongNoFrameskip-v4
tensorboard --logdir runs
```

<!-- # Errors
This system does not have apparently enough memory to store the complete replay buffer 28.24GB > 8.13GB -->

# Requirements

Requirements pulled from `https://github.com/vwxyzjn/cleanrl`.
```
absl-py==1.4.0 ; python_version >= "3.8" and python_version < "3.11"
ale-py==0.8.1 ; python_version >= "3.8" and python_version < "3.11"
appdirs==1.4.4 ; python_version >= "3.8" and python_version < "3.11"
autorom-accept-rom-license==0.6.1 ; python_version >= "3.8" and python_version < "3.11"
autorom[accept-rom-license]==0.4.2 ; python_version >= "3.8" and python_version < "3.11"
cachetools==5.3.0 ; python_version >= "3.8" and python_version < "3.11"
certifi==2023.5.7 ; python_version >= "3.8" and python_version < "3.11"
charset-normalizer==3.1.0 ; python_version >= "3.8" and python_version < "3.11"
click==8.1.3 ; python_version >= "3.8" and python_version < "3.11"
cloudpickle==2.2.1 ; python_version >= "3.8" and python_version < "3.11"
colorama==0.4.4 ; python_version >= "3.8" and python_version < "3.11"
commonmark==0.9.1 ; python_version >= "3.8" and python_version < "3.11"
cycler==0.11.0 ; python_version >= "3.8" and python_version < "3.11"
decorator==4.4.2 ; python_version >= "3.8" and python_version < "3.11"
docker-pycreds==0.4.0 ; python_version >= "3.8" and python_version < "3.11"
docstring-parser==0.15 ; python_version >= "3.8" and python_version < "3.11"
farama-notifications==0.0.4 ; python_version >= "3.8" and python_version < "3.11"
filelock==3.12.0 ; python_version >= "3.8" and python_version < "3.11"
fonttools==4.38.0 ; python_version >= "3.8" and python_version < "3.11"
gitdb==4.0.10 ; python_version >= "3.8" and python_version < "3.11"
gitpython==3.1.31 ; python_version >= "3.8" and python_version < "3.11"
google-auth-oauthlib==0.4.6 ; python_version >= "3.8" and python_version < "3.11"
google-auth==2.18.0 ; python_version >= "3.8" and python_version < "3.11"
grpcio==1.54.0 ; python_version >= "3.8" and python_version < "3.11"
gym-notices==0.0.8 ; python_version >= "3.8" and python_version < "3.11"
gym==0.23.1 ; python_version >= "3.8" and python_version < "3.11"
gymnasium==0.28.1 ; python_version >= "3.8" and python_version < "3.11"
huggingface-hub==0.11.1 ; python_version >= "3.8" and python_version < "3.11"
idna==3.4 ; python_version >= "3.8" and python_version < "3.11"
imageio-ffmpeg==0.3.0 ; python_version >= "3.8" and python_version < "3.11"
imageio==2.28.1 ; python_version >= "3.8" and python_version < "3.11"
importlib-metadata==5.2.0 ; python_version >= "3.8" and python_version < "3.10"
importlib-resources==5.12.0 ; python_version >= "3.8" and python_version < "3.11"
jax-jumpy==1.0.0 ; python_version >= "3.8" and python_version < "3.11"
kiwisolver==1.4.4 ; python_version >= "3.8" and python_version < "3.11"
markdown==3.3.7 ; python_version >= "3.8" and python_version < "3.11"
markupsafe==2.1.2 ; python_version >= "3.8" and python_version < "3.11"
matplotlib==3.5.3 ; python_version >= "3.8" and python_version < "3.11"
moviepy==1.0.3 ; python_version >= "3.8" and python_version < "3.11"
numpy==1.24.4 ; python_version >= "3.8" and python_version < "3.11"
oauthlib==3.2.2 ; python_version >= "3.8" and python_version < "3.11"
opencv-python==4.7.0.72 ; python_version >= "3.8" and python_version < "3.11"
packaging==23.1 ; python_version >= "3.8" and python_version < "3.11"
pandas==1.3.5 ; python_version >= "3.8" and python_version < "3.11"
pathtools==0.1.2 ; python_version >= "3.8" and python_version < "3.11"
pillow==9.5.0 ; python_version >= "3.8" and python_version < "3.11"
proglog==0.1.10 ; python_version >= "3.8" and python_version < "3.11"
protobuf==3.19.0 ; python_version < "3.11" and python_version >= "3.8"
psutil==5.9.5 ; python_version >= "3.8" and python_version < "3.11"
pyasn1-modules==0.3.0 ; python_version >= "3.8" and python_version < "3.11"
pyasn1==0.5.0 ; python_version >= "3.8" and python_version < "3.11"
pygame==2.1.0 ; python_version >= "3.8" and python_version < "3.11"
pygments==2.15.1 ; python_version >= "3.8" and python_version < "3.11"
pyparsing==3.0.9 ; python_version >= "3.8" and python_version < "3.11"
python-dateutil==2.8.2 ; python_version >= "3.8" and python_version < "3.11"
pytz==2023.3 ; python_version >= "3.8" and python_version < "3.11"
pyyaml==6.0.1 ; python_version >= "3.8" and python_version < "3.11"
requests-oauthlib==1.3.1 ; python_version >= "3.8" and python_version < "3.11"
requests==2.30.0 ; python_version >= "3.8" and python_version < "3.11"
rich==11.2.0 ; python_version >= "3.8" and python_version < "3.11"
rsa==4.7.2 ; python_version >= "3.8" and python_version < "3.11"
sentry-sdk==1.22.2 ; python_version >= "3.8" and python_version < "3.11"
setproctitle==1.3.2 ; python_version >= "3.8" and python_version < "3.11"
setuptools==67.7.2 ; python_version >= "3.8" and python_version < "3.11"
shimmy==1.1.0 ; python_version >= "3.8" and python_version < "3.11"
shtab==1.6.4 ; python_version >= "3.8" and python_version < "3.11"
six==1.16.0 ; python_version >= "3.8" and python_version < "3.11"
smmap==5.0.0 ; python_version >= "3.8" and python_version < "3.11"
stable-baselines3==2.0.0 ; python_version >= "3.8" and python_version < "3.11"
tenacity==8.2.3 ; python_version >= "3.8" and python_version < "3.11"
tensorboard-data-server==0.6.1 ; python_version >= "3.8" and python_version < "3.11"
tensorboard-plugin-wit==1.8.1 ; python_version >= "3.8" and python_version < "3.11"
tensorboard==2.11.2 ; python_version >= "3.8" and python_version < "3.11"
torch==1.12.1 ; python_version >= "3.8" and python_version < "3.11"
tqdm==4.65.0 ; python_version >= "3.8" and python_version < "3.11"
typing-extensions==4.5.0 ; python_version >= "3.8" and python_version < "3.11"
tyro==0.5.10 ; python_version >= "3.8" and python_version < "3.11"
urllib3==1.26.15 ; python_version >= "3.8" and python_version < "3.11"
wandb==0.13.11 ; python_version >= "3.8" and python_version < "3.11"
werkzeug==2.2.3 ; python_version >= "3.8" and python_version < "3.11"
wheel==0.40.0 ; python_version >= "3.8" and python_version < "3.11"
zipp==3.15.0 ; python_version >= "3.8" and python_version < "3.10"
```

