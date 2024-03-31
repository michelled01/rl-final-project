import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tf_agents.networks import q_network # dqn impl from tensorflow

class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialEmbedding, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)                                                 
        return x

class VisionBackbone(nn.Module):
    def __init__(self, backbone_type):
        super(VisionBackbone, self).__init__()
        if backbone_type == 'Impala-CNN':
            # TODO: impl impala-cnn architecture
            pass
        elif backbone_type == 'ResNet':
            # TODO: impl resnet architecture
            pass

class QNetwork(q_network.QNetwork):
    def __init__(self,
        input_tensor_spec,
        action_spec,
        num_actions_per_game,
        num_games,
        fc_layer_params=(100,),
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform',
        name='MultiHeadQNetwork'):
        
        super(QNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            last_kernel_initializer=last_kernel_initializer,
            name=name)
        
        self.num_actions_per_game = num_actions_per_game
        self.num_games = num_games

        # Create linear projection layers for each game
        self.heads = [tf.keras.layers.Dense(num_actions_per_game, kernel_initializer=kernel_initializer) 
                      for _ in range(num_games)]

    def call(self, inputs, step_type=None, network_state=()):
        # Forward pass through the base QNetwork
        base_output = super(QNetwork, self).call(inputs, step_type, network_state)

        # Apply linear projection layers for each game
        q_values_per_game = [head(base_output) for head in self.heads]

        return q_values_per_game
    
        # TODO: set up distributional rewards