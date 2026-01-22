# Flappy-Bird-Project

# ITIS Project: Flappy Bird AI

**Group:** Sebastian Balling, Jonas Krüger, Victor Holt  
**Date:** January 2026

## About the Project
This repository contains the code and data for our project on genetic algorithms and elite selection. Further details on the project can be read in the report.

## File Overview
* **`agent_plays.py`**: The main script where the AI is created and trained.
* **`Flappy_bird_Game.py`**: Our own version of the game itself, including the game physics and environment.
* **`run_experiment.py`**: A script we wrote to run the full experiment automatically. It loops through Elite 1, 3, 5 and inserts the data into seperate csv files.
* **`sample_size.py`**: This is the script used for the statistical power analysis (Cohen's f) to determine our sample size ($n=16$).
* **`play_yourself.py`**: A manual version of the game if you want to test the physics yourself. This was used for brainstorming in the early stages of the project.
* **`Endelig_flappy_data_fil.xlsx`**: The raw data output from our experiments gathered in a single xlsx file. When the file run_experiment.py is run, the data will be gathered in three seperate csv files.

## Setup
It is needed to have Python installed to run the code. The following libraries were used:

```bash
pip install torch pygame pandas numpy statsmodels openpyxl
