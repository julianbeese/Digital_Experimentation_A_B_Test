import numpy as np
import scipy.stats as stats


def calculate_confidence_interval(successes, total, confidence=0.95):
    """
    Calculates the confidence interval for a proportion using the Wilson Score method.
    """
    if total == 0:
        return 0, 0, 0

    proportion = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)

    # Wilson Score Interval
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)) / denominator

    lower_bound = max(0, center - margin)
    upper_bound = min(1, center + margin)

    return proportion, lower_bound, upper_bound


def run_ab_test(data_a, data_b, metric_name, min_effect_size=0.05, confidence=0.95):
    """
    Performs an A/B test and returns the results.
    data_a: Dictionary with 'success' and 'total' for variant A
    data_b: Dictionary with 'success' and 'total' for variant B
    metric_name: Name of the metric (e.g., 'CTR' or 'Bounce Rate')
    min_effect_size: Minimum effect size to be considered significant
    confidence: Confidence level (default: 0.95)
    """
    # Calculate conversion rates and confidence intervals
    rate_a, ci_lower_a, ci_upper_a = calculate_confidence_interval(data_a['success'], data_a['total'], confidence)
    rate_b, ci_lower_b, ci_upper_b = calculate_confidence_interval(data_b['success'], data_b['total'], confidence)

    # Calculate relative change (positive for CTR, negative for Bounce Rate)
    if metric_name == "CTR":
        relative_change = (rate_b - rate_a) / rate_a
        significant_improvement = rate_b > rate_a * (1 + min_effect_size) and ci_lower_b > ci_upper_a
    elif metric_name == "Bounce Rate":
        relative_change = (rate_a - rate_b) / rate_a  # For Bounce Rate, a reduction is better
        significant_improvement = rate_b < rate_a * (1 - min_effect_size) and ci_upper_b < ci_lower_a

    # Calculate standard deviation for both variants
    # For proportions: Standard deviation = sqrt(p * (1-p))
    std_dev_a = np.sqrt(rate_a * (1 - rate_a))
    std_dev_b = np.sqrt(rate_b * (1 - rate_b))

    # Calculate Z-score for the difference
    pooled_std = np.sqrt(std_dev_a ** 2 / data_a['total'] + std_dev_b ** 2 / data_b['total'])
    z_score = (rate_b - rate_a) / pooled_std if pooled_std > 0 else 0

    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Create result dictionary
    result = {
        'metric': metric_name,
        'A_name': "Applications Open Now",
        'B_name': "Applications Closing Soon",
        'A_rate': rate_a,
        'A_ci_lower': ci_lower_a,
        'A_ci_upper': ci_upper_a,
        'A_sample_size': data_a['total'],
        'B_rate': rate_b,
        'B_ci_lower': ci_lower_b,
        'B_ci_upper': ci_upper_b,
        'B_sample_size': data_b['total'],
        'relative_change': relative_change,
        'min_effect_size': min_effect_size,
        'confidence': confidence,
        'z_score': z_score,
        'p_value': p_value,
        'significant_improvement': significant_improvement
    }

    return result


def main():
    # Real A/B test data

    sample_size = 1090

    # CTR (Click-Through Rate) data
    ctr_a_percent = 24.88  # 24.88% CTR for "Applications Open Now"
    ctr_b_percent = 31.24  # 31.24% CTR for "Applications Closing Soon"

    # Convert to absolute numbers
    ctr_data_a = {'success': int(sample_size * ctr_a_percent / 100), 'total': sample_size}
    ctr_data_b = {'success': int(sample_size * ctr_b_percent / 100), 'total': sample_size}

    # Bounce Rate data (lower is better)
    bounce_a_percent = 62.81  # 62.81% Bounce Rate for "Applications Open Now"
    bounce_b_percent = 65.00  # 65.00% Bounce Rate for "Applications Closing Soon"

    # Convert to absolute numbers
    bounce_data_a = {'success': int(sample_size * bounce_a_percent / 100), 'total': sample_size}
    bounce_data_b = {'success': int(sample_size * bounce_b_percent / 100), 'total': sample_size}

    alpha = 0.05

    # Run tests
    ctr_result = run_ab_test(ctr_data_a, ctr_data_b, "CTR", min_effect_size=0.05, confidence=1 - alpha)
    bounce_result = run_ab_test(bounce_data_a, bounce_data_b, "Bounce Rate", min_effect_size=0.05, confidence=1 - alpha)

    # Collect all results
    results = [ctr_result, bounce_result]

    for result in results:
        print(f"\nResults for {result['metric']}:")
        conf_level = result['confidence'] * 100
        print(f"Confidence level: {conf_level:.1f}%")
        print(f"A ({result['A_name']}): {result['A_rate']:.4f}")
        print(f"  {conf_level:.1f}% Confidence Interval: [{result['A_ci_lower']:.4f}, {result['A_ci_upper']:.4f}]")
        print(f"B ({result['B_name']}): {result['B_rate']:.4f}")
        print(f"  {conf_level:.1f}% Confidence Interval: [{result['B_ci_lower']:.4f}, {result['B_ci_upper']:.4f}]")
        print(f"Relative change: {result['relative_change'] * 100:.2f}%")
        print(f"p-value: {result['p_value']:.6f}")
        print(f"Alpha (significance level): {alpha:.6f}")
        print(f"Statistically significant: {'Yes' if result['p_value'] < alpha else 'No'}")
        print(
            f"Significant improvement (incl. minimum effect size): {'Yes' if result['significant_improvement'] else 'No'}")


if __name__ == "__main__":
    main()