# Flappy-Bird-Project

# Flappy Bird AI - Genetic Algorithms and Elite Selection

**Authors:** Sebastian Balling, Jonas Krüger, and Victor Holt  
**Project:** ITIS Semester Project  
**Date:** January 2026  

## Project Overview
This repository contains the source code and data for our research project investigating the impact of **Elite Selection** size on the performance of a Genetic Algorithm (GA) training a neural network to play Flappy Bird.

The project compares three specific elite configurations:
- **Top 1:** Only the single best agent reproduces.
- **Top 3:** The top 3 agents reproduce.
- **Top 5:** The top 5 agents reproduce.

The AI is implemented using **PyTorch** (Neural Network) and a custom Genetic Algorithm loop without crossover (mutation-only).

## Repository Structure

### Core Logic
* **`agent_plays.py`**: The main training script. It runs the genetic algorithm, initializes the population, handles mutation/selection, and logs performance data.
* **`Flappy_bird_Game.py`**: The game engine. Contains the `Bird` and `Tunnel` classes, physics logic, and collision detection.
* **`run_experiment.py`**: A wrapper script that automates the full experiment by running `agent_plays.py` sequentially for elite sizes 1, 3, and 5.

### Analysis & Tools
* **`sample_size.py`**: A statistical script that calculates Cohen's *f* (Effect Size) and performs an ANOVA Power analysis to determine the required sample size ($n=16$).
* **`play_yourself.py`**: A manual version of the game that allows a human to play using the spacebar/mouse. Useful for understanding the game physics.

### Data
* **`Endelig_flappy_data_fil(1).xlsx - *.csv`**: These CSV files contain the raw data from our experimental runs (Generation, Average Score, Best Fitness, etc.).

---

## Installation & Requirements

To run the code, you need Python installed along with the following libraries:

```bash
pip install torch pygame pandas numpy statsmodels openpyxl
