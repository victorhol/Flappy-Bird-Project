import os

runs = 16
generations = 25

top_agents = [1, 3, 5]

for n in scenarios:
    print(f"---Top {n}---")
    
    cmd = (f"python agent_plays.py "
           f"--runs {runs} "
           f"--generations {generations} "
           f"--elite_n {n} "
           f"--summary_csv data_top{n}.csv ")
    
    os.system(cmd)

print("\nThe eksperiment is done. The data can be found in the data_topN.csv file(s)")
