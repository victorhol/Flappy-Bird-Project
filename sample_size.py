import numpy as np
from statsmodels.stats.power import FTestAnovaPower

# --- 1. INPUT DATA FRA PILOTSTUDIET (Generation 25) ---
# Gennemsnit (Mean) og Standardafvigelse (SD) for de 3 grupper
# Top 1: Mean=388.33, SD=480.95
# Top 3: Mean=886.33, SD=195.14
# Top 5: Mean=999.00, SD=0.00
means = np.array([388.33, 886.33, 999.00])
sds = np.array([480.95, 195.14, 0.00])
n_pilot = 3      # Antal runs i pilotstudiet
k_groups = 3     # Antal grupper (Top 1, Top 3, Top 5)

# --- 2. EFFEKTSTØRRELSEN (COHEN'S F) ---

# A. Vi beregner 'Between-Group Variability' (Spredning mellem grupperne)
# Dette er standardafvigelsen af de tre gennemsnit
sigma_m = np.std(means, ddof=0) 

# B. Vi beregner 'Pooled Standard Deviation' (Gennemsnitlig spredning indeni grupperne)
# Formel: sqrt( sum((n-1)*s^2) / (N_total - k) )
variances = sds**2
numerator = np.sum((n_pilot - 1) * variances)
denominator = (n_pilot * k_groups) - k_groups
pooled_sd = np.sqrt(numerator / denominator)

# C. Beregn Cohen's f
# Formel: f = sigma_m / pooled_sd
cohens_f = sigma_m / pooled_sd

print(f"Effect Size (Cohen's f): {cohens_f:.4f}")
print(f"Pooled SD: {pooled_sd:.2f}")

# --- 3. POWER ANALYSIS (VI FINDER N) ---
# Vi bruger F-test for ANOVA
alpha = 0.05    # Signifikansniveau (95% sikkerhed)
power = 0.80    # Statistisk styrke (80% chance for at finde forskel)

analysis = FTestAnovaPower()
required_n = analysis.solve_power(
    effect_size=cohens_f, 
    alpha=alpha, 
    power=power, 
    k_groups=k_groups
)

print(f"Den estimerede sample size (n) er: {required_n:.4f}")
