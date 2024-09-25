# PongAi

Recreated Pong using PyGame library and employed deep reinforcement learning techniques to train a 2D CNN using TensorFlow and OpenCV to play against itself

# Deep Q-Network (DQN) Pong Game Implementation

## Overview

This project develops a Pong game agent using **Deep Q-Networks (DQN)**, a type of reinforcement learning algorithm. The agent learns optimal strategies through continuous interaction and trial-and-error within the game environment.

## Components

- **Pong Game Environment**: Created with `pygame`, it features a basic rendition of Pong with two paddles and a ball, allowing state updates via player actions.
- **Deep Q-Network Agent**: Utilizes `TensorFlow` to learn the game by estimating the Q-function, which predicts potential rewards from actions in specific game states.

## Q-Learning and Deep Q-Networks

### Q-Learning

A model-free reinforcement learning algorithm that aims to determine the value of an action in a particular state using the Bellman Equation: 


Where:
- `Q(s, a)` is the expected reward for an action `a` in state `s`,
- `r` is the immediate reward,
- `gamma` is the discount factor,
- `s'` is the next state, and
- `a'` are potential next actions.

### Deep Q-Networks (DQN)

DQN integrates Q-Learning with deep neural networks to manage environments with complex state spaces like video game inputs. It uses a neural network to approximate the Q-table, with key features like:
- **Experience Replay**: Stores past experiences and samples random minibatches to break correlations between sequential data.
- **Fixed Target Network**: Uses a separate network to calculate target Q-values, reducing oscillations during training.

## How the Agent Learns

### Neural Network Architecture

- **Convolutional Layers**: Process visual input to extract features.
  - Three layers with varying filter sizes and strides.
- **Fully Connected Layers**: Determine action values.
  - Layers include a dense layer and an output layer corresponding to action choices.

### Training Process

1. **Initialization**: Starts with random weights and an initial game state.
2. **Action Selection**: Uses an epsilon-greedy policy for a balance between exploration and exploitation, with epsilon gradually decreasing.
3. **Experience Replay**: Stores transitions and samples them randomly to reduce correlation and stabilize updates.
4. **Minibatch Training**: Uses sampled experiences to update network weights based on the Bellman Equation.

### Learning Flow

- **Observation**: Agent assesses the current state from game frames.
- **Decision Making**: Chooses an action using the policy.
- **Environment Interaction**: Executes the action and receives new state and reward.
- **Memory Storage**: Captures experience in replay memory.
- **Training**: Periodically trains on a batch from memory.
