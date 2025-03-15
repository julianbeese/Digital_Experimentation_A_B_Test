import numpy as np
from scipy import stats
import math

# Parameters
mu_0 = 0.10  # baseline conversion rate
mu_1 = 0.15  # new conversion rate
sigma = math.sqrt(mu_0 * (1 - mu_0))  # standard deviation
effect_size = (mu_1 - mu_0) / sigma

# Desired power and significance level
power = 0.80
alpha = 0.05

# Z-values for alpha and power
z_alpha_2 = stats.norm.ppf(1 - (alpha / 2))  # two-sided test
z_beta = stats.norm.ppf(power)

# Calculate sample size using the formula
n = ((z_alpha_2 + z_beta)**2 * 2 * sigma**2) / effect_size**2

# Round up since you can't have a fraction of a sample
n = math.ceil(n)

# Print required sample size per group
print("Required sample size per group:", n)






