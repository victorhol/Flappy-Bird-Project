import os

# Konfiguration
runs = 3
generations = 35

# Vi vil teste disse 3 scenarier
scenarios = [3, 5]  # Top 1, Top 3, Top 5

print("Starter eksperimentet...")

for n in scenarios:
    print(f"\n--- Starter simulering for Top {n} ---")
    
    # Vi giver filen et unikt navn, f.eks. "summary_top1.csv"
    cmd = (f"python opdateret_sim.py "
           f"--runs {runs} "
           f"--generations {generations} "
           f"--elite_n {n} "
           f"--summary_csv data_top{n}.csv "
           f"--details_csv details_top{n}.csv")
    
    os.system(cmd)

print("\nAlle eksperimenter er færdige! Data ligger i data_top1.csv, data_top3.csv, osv.")