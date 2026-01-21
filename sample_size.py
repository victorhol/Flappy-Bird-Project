import numpy as np
from statsmodels.stats.power import FTestAnovaPower


means = np.array([388.33, 886.33, 999.00])
sds = np.array([480.95, 195.14, 0.00])
n_pilot = 3
k_groups = 3

sigma_m = np.std(means, ddof=0) 

variances = sds**2
numerator = np.sum((n_pilot - 1) * variances)
denominator = (n_pilot * k_groups) - k_groups
pooled_sd = np.sqrt(numerator / denominator)

cohens_f = sigma_m / pooled_sd

print(f"Effect Size (Cohen's f): {cohens_f:.4f}")

alpha = 0.05
power = 0.80

analysis = FTestAnovaPower()
required_n = analysis.solve_power(
    effect_size=cohens_f, 
    alpha=alpha, 
    power=power, 
    k_groups=k_groups
)

print(f"Required Sample Size per Group (n): {required_n:.4f}")