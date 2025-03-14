import math

# Given values
baseline_time_on_page = 6.00  # Seconds (average from the data)
min_detectable_effect = 0.05  # 5% change
target_time_on_page = baseline_time_on_page * (1 + min_detectable_effect)  # Target value
std_dev = 8.80  # Standard deviation from the data
alpha = 0.05  # Significance level
power = 0.8  # Desired statistical power
z_alpha = 1.96  # Z-value for alpha/2 = 0.025 (two-tailed test)
z_beta = 0.84  # Z-value for power = 0.8

# Calculation of effect size
delta = target_time_on_page - baseline_time_on_page
effect_size = delta / std_dev

# Calculation of required sample size per group
samples_per_group = math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2)

# Total sample size
total_sample_size = samples_per_group * 2

# Output of results
print("Power calculation for A/B test (Time on Page):")
print("------------------------------------------")
print(f"Baseline (Version A): {baseline_time_on_page:.2f} seconds")
print(f"Target value (Version B): {target_time_on_page:.2f} seconds ({min_detectable_effect*100}% increase)")
print(f"Standard deviation: {std_dev:.2f} seconds")
print(f"Significance level (α): {alpha}")
print(f"Statistical power (1-β): {power}")
print(f"δ (Difference): {delta:.2f} seconds")
print(f"Effect size (d): {effect_size:.4f}")
print(f"Calculation: n = ((Z₁₋ₐ/₂ + Z₁₋ᵦ)² · σ²) / δ²")
print(f"n = (({z_alpha} + {z_beta})² · {std_dev}²) / {delta}²")
print(f"n = ({(z_alpha + z_beta)**2:.2f} · {std_dev**2:.2f}) / {delta**2:.2f}")
print(f"n = {((z_alpha + z_beta)**2 * std_dev**2):.2f} / {delta**2:.2f}")
print(f"n = {((z_alpha + z_beta)**2 * std_dev**2) / delta**2:.2f}")
print(f"Required sample size per group: {samples_per_group}")
print(f"Total required sample size: {total_sample_size}")