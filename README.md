# AlphaZero Gobang

This project is a **Gobang (五子棋)** game built using **Reinforcement Learning** powered by the **AlphaZero** algorithm and **Monte Carlo Tree Search (MCTS)**. It includes a **PyQt5 interface** for human vs AI gameplay and uses **PyTorch** for training the policy-value network.

## Features

- **Reinforcement Learning**: Trained using AlphaZero-style self-play.
- **Monte Carlo Tree Search (MCTS)**: AI decision-making based on MCTS.
- **PyQt5 GUI**: Human vs AI gameplay with a simple and intuitive interface.

## Requirements

- Python 3.8+
- PyTorch (for model training)
- PyQt5 (for GUI)
- Other dependencies listed in `requirements.txt`

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the repository:
```bash
    git clone https://github.com/your-username/AlphaZero-Gobang.git
```
### 2. Set up a virtual environment:

It is recommended to create a virtual environment for Python dependencies.
```bash
    # Create a virtual environment using venv
    python -m venv venv
    # or using conda
    conda create -n gomoku python=3.8
    conda activate gomoku
```
### 3. Install dependencies:

Install the required Python dependencies by running:
```bash
    pip install -r requirements.txt
```
## Usage
### 1. Train the model (Self-Play Mode)

To train the model using self-play and Monte Carlo Tree Search:
```bash
    python train.py --board-width 6 --board-height 6 --n-in-row 4 --n-playout 100 --game-batch-num 40 --checkpoint-freq 20 --save-dir ./checkpoints
```
### 2. Play against the AI (PyQt Interface)

    After training, you can play against the trained AI using the following command:
```bash
    python play_pyqt.py --model ./checkpoints/best_model.pt --board-width 6 --board-height 6 --n-in-row 4
```
## Project Structure

    train.py: Script for training the model using self-play and saving the best model.
    
    play_pyqt.py: PyQt5 interface for playing against the AI.
    
    gomoku/env.py: Custom environment for the game, built using the Gym API.
    
    gomoku/mcts.py: Implementation of the Monte Carlo Tree Search (MCTS) for decision-making.
    
    gomoku/policy_value_net.py: Neural network model using PyTorch for policy-value estimation.

## License

    This project is licensed under the MIT License - see the LICENSE
     file for details.

## Contributing

    Contributions are welcome! Please feel free to fork the repository, create a branch, and submit a pull request. When contributing, please follow the typical GitHub workflow:
    
    Fork the repository.
    
    Clone your fork locally.
    
    Create a new branch for your feature or bug fix.
    
    Make your changes and commit them.
    
    Push your changes and create a pull request.
    
    If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.