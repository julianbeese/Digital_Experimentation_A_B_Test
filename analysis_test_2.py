import pandas as pd
import numpy as np
from scipy import stats
import math
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Function to analyze A/B test results
def analyze_ab_test(file_path, start_date="2025-03-05", alpha=0.05):
    """
    Analyze A/B test results from an Excel file with date columns,
    only considering data from the start_date onwards

    Parameters:
    file_path (str): Path to the Excel file
    start_date (str): Start date of the test in YYYY-MM-DD format
    alpha (float): Significance level, default is 0.05

    Returns:
    dict: Analysis results
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully read the Excel file. Shape: {df.shape}")

        # Extract the first column which might contain group labels
        first_col_name = df.columns[0]
        print(f"First column name: {first_col_name}")
        print(f"Values in first column: {df[first_col_name].tolist()}")

        # Convert start_date to datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        print(f"Using data from {start_date} onwards")

        # Filter columns based on the date
        date_columns = [col for col in df.columns[1:] if isinstance(col, datetime) and col >= start_date]
        print(f"Selected date columns: {date_columns}")

        if not date_columns:
            raise ValueError(f"No data columns found after the start date {start_date}")

        # Extract data for Group A and Group B, only for the selected date columns
        group_a_row = df.iloc[0]
        group_b_row = df.iloc[1]

        # Get data only from the selected date columns
        group_a_data = pd.to_numeric(group_a_row[date_columns], errors='coerce').dropna()
        group_b_data = pd.to_numeric(group_b_row[date_columns], errors='coerce').dropna()

        print(f"Group A identifier: {group_a_row.iloc[0]}")
        print(f"Group B identifier: {group_b_row.iloc[0]}")
        print(f"Group A size (number of days): {len(group_a_data)}")
        print(f"Group B size (number of days): {len(group_b_data)}")

        # Calculate descriptive statistics
        group_a_mean = group_a_data.mean()
        group_b_mean = group_b_data.mean()
        group_a_std = group_a_data.std(ddof=1)  # Using n-1 for sample standard deviation
        group_b_std = group_b_data.std(ddof=1)  # Using n-1 for sample standard deviation

        print(f"Group A mean: {group_a_mean:.2f}, std: {group_a_std:.2f}")
        print(f"Group B mean: {group_b_mean:.2f}, std: {group_b_std:.2f}")

        # Calculate relative change
        rel_change = (group_b_mean - group_a_mean) / group_a_mean * 100

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=False)

        # Calculate 95% confidence interval
        # For the difference between means
        n1, n2 = len(group_a_data), len(group_b_data)
        dof = (group_a_std ** 2 / n1 + group_b_std ** 2 / n2) ** 2 / (
                    (group_a_std ** 2 / n1) ** 2 / (n1 - 1) + (group_b_std ** 2 / n2) ** 2 / (n2 - 1))
        t_critical = stats.t.ppf(1 - alpha / 2, dof)

        std_err = math.sqrt(group_a_std ** 2 / n1 + group_b_std ** 2 / n2)
        margin_of_error = t_critical * std_err

        ci_lower = (group_b_mean - group_a_mean) - margin_of_error
        ci_upper = (group_b_mean - group_a_mean) + margin_of_error

        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(((n1 - 1) * group_a_std ** 2 + (n2 - 1) * group_b_std ** 2) / (n1 + n2 - 2))
        cohens_d = (group_b_mean - group_a_mean) / pooled_std

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # Calculate power
        analysis = TTestIndPower()
        power = analysis.power(effect_size=abs(cohens_d), nobs1=n1, alpha=alpha, ratio=n2 / n1)

        # Create results dictionary
        results = {
            "group_a_mean": group_a_mean,
            "group_b_mean": group_b_mean,
            "group_a_std": group_a_std,
            "group_b_std": group_b_std,
            "relative_change_percent": rel_change,
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_interval": (ci_lower, ci_upper),
            "effect_size": cohens_d,
            "effect_interpretation": effect_interpretation,
            "is_significant": p_value < alpha,
            "statistical_power": power
        }

        # Print results
        print("\n===== A/B Test Analysis Results =====")
        print(f"Analysis period: {min(date_columns).strftime('%Y-%m-%d')} to {max(date_columns).strftime('%Y-%m-%d')}")
        print(f"Baseline (Group A): {group_a_mean:.2f} seconds")
        print(f"Treatment (Group B): {group_b_mean:.2f} seconds")
        print(f"Absolute change: {group_b_mean - group_a_mean:.2f} seconds")
        print(f"Relative change: {rel_change:.2f}%")
        print(f"95% Confidence Interval for mean difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Statistically significant (α={alpha})? {'Yes' if p_value < alpha else 'No'}")
        print(f"Cohen's d effect size: {cohens_d:.3f} ({effect_interpretation})")
        print(f"Statistical power: {power:.3f}")

        if power < 0.8:
            print("Warning: Statistical power is below the recommended 0.8 threshold.")

        # Visualize results
        plt.figure(figsize=(12, 8))

        # Time series plot
        plt.subplot(2, 1, 1)
        dates_str = [d.strftime('%m-%d') for d in date_columns]
        plt.plot(dates_str, group_a_data.values, 'o-', label=f'Group A ({group_a_row.iloc[0]})')
        plt.plot(dates_str, group_b_data.values, 's-', label=f'Group B ({group_b_row.iloc[0]})')
        plt.xlabel('Date')
        plt.ylabel('Time on Page (seconds)')
        plt.title('Daily Time on Page by Group')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Bar chart comparison with error bars
        plt.subplot(2, 2, 3)
        bars = plt.bar(['Group A', 'Group B'], [group_a_mean, group_b_mean],
                       yerr=[group_a_std / math.sqrt(n1), group_b_std / math.sqrt(n2)],
                       capsize=10, alpha=0.7, color=['lightblue', 'lightgreen'])
        plt.ylabel('Mean Time on Page (seconds)')
        plt.title('Mean Time on Page with 95% CI')

        # Add percentage change annotation
        max_height = max(group_a_mean, group_b_mean) + max(group_a_std / math.sqrt(n1), group_b_std / math.sqrt(n2))
        plt.annotate(f'{rel_change:.1f}%',
                     xy=(1, group_b_mean),
                     xytext=(1.25, group_b_mean + max_height * 0.1),
                     arrowprops=dict(arrowstyle='->'))

        # Boxplot for distribution comparison
        plt.subplot(2, 2, 4)
        boxplot = plt.boxplot([group_a_data, group_b_data], labels=['Group A', 'Group B'], patch_artist=True)
        plt.ylabel('Time on Page (seconds)')
        plt.title('Distribution of Time on Page')

        # Set boxplot colors
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)

        plt.tight_layout()
        plt.savefig('ab_test_results.png', dpi=300)
        print("Visualization saved as 'ab_test_results.png'")

        return results

    except Exception as e:
        print(f"Error analyzing the data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Call the function with the uploaded Excel file
# Using March 5, 2025 as the start date for the test
results = analyze_ab_test('test_2_time_on_page.xlsx', start_date="2025-03-05")

# Required sample size calculation for future tests
if results:
    # Calculate required sample size for 5% increase with 80% power
    baseline_mean = results["group_a_mean"]
    std_dev = results["group_a_std"]
    effect_size = 0.05 * baseline_mean / std_dev  # 5% increase in standardized form

    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=0.8, alpha=0.05)

    print("\n===== Sample Size Calculation for Future Tests =====")
    print(f"To detect a 5% increase with 80% power at α=0.05:")
    print(f"Required sample size per group: {math.ceil(sample_size)}")
    print(f"Total required sample size: {math.ceil(sample_size * 2)}")

    # Formula-based calculation for comparison
    z_alpha = 1.96  # for α=0.05
    z_beta = 0.84  # for power=0.8
    delta = 0.05 * baseline_mean
    sample_size_formula = math.ceil(2 * ((z_alpha + z_beta) ** 2 * std_dev ** 2) / delta ** 2)

    print(f"Sample size using formula: {sample_size_formula} per group")
    print(f"Total sample size using formula: {sample_size_formula * 2}")