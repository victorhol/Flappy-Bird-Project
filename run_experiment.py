import os

runs = 1
generations = 5

top_agents = [1, 3, 5]

for n in top_agents:
    print(f"---Top {n}---")
    
    cmd = (f"python agent_plays.py "
           f"--runs {runs} "
           f"--generations {generations} "
           f"--elite_n {n} "
           f"--summary_csv data_top{n}.csv ")
    
    os.system(cmd)

print("\nThe eksperiment is done. Data in data_top1.csv, data_top3.csv and data_top5.csv")
