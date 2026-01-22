# Flappy-Bird-Project

# ITIS Semester Project: Flappy Bird AI

**Group:** Sebastian Balling, Jonas Krüger, Victor Holt  
**Date:** January 2026

## About the Project
This repository contains the code and data for our project on Genetic Algorithms. We built a Neuroevolution system to train an agent to play Flappy Bird from scratch. 

The main goal was to investigate how **Elite Selection** affects learning stability. Specifically, we compared keeping the Top 1, Top 3, and Top 5 agents between generations.

## File Overview

Here is a quick breakdown of the files in the repo:

* **`agent_plays.py`**: The main script. This runs the actual training loop, handles the neural network, mutation, and logging.
* **`Flappy_bird_Game.py`**: Our implementation of the game itself (physics, collisions, pipes).
* **`run_experiment.py`**: A helper script we wrote to run the full experiment automatically (it loops through Elite 1, 3, and 5 sequentially).
* **`sample_size.py`**: Script used for the statistical power analysis (Cohen's f) to determine our sample size ($n=16$).
* **`play_yourself.py`**: A manual version of the game if you want to test the physics yourself.
* **`data_topN.csv`**: The raw data output from our experiments.

## Setup
You need Python installed. We used the following libraries:

```bash
pip install torch pygame pandas numpy statsmodels openpyxl
