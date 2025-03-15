# Import necessary libraries
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
ab_test_data = pd.read_csv("ab_test_data.csv")

# Difference in Means Estimator for Dwell Time between Group A and Group B
ab_subset = ab_test_data[ab_test_data['Group'].isin(['A', 'B'])]
a_data = ab_subset[ab_subset['Group'] == 'A']['Dwell_Time']
b_data = ab_subset[ab_subset['Group'] == 'B']['Dwell_Time']
a_b_means = stats.ttest_ind(a_data, b_data)

# Difference in Means Estimator for Dwell Time between Group A and Group C
ac_subset = ab_test_data[ab_test_data['Group'].isin(['A', 'C'])]
a_data = ac_subset[ac_subset['Group'] == 'A']['Dwell_Time']
c_data = ac_subset[ac_subset['Group'] == 'C']['Dwell_Time']
a_c_means = stats.ttest_ind(a_data, c_data)

# Output the results of difference in means tests
print("A vs B t-test:", a_b_means)
print("A vs C t-test:", a_c_means)

# Regression Analysis: Dwell Time on Group Membership
# Convert Group to a categorical variable for regression
ab_test_data['Group'] = pd.Categorical(ab_test_data['Group'])

# Create dummy variables for the Group categorical variable
dummy_vars = pd.get_dummies(ab_test_data['Group'], prefix='Group', drop_first=True)
regression_data = pd.concat([ab_test_data[['Dwell_Time', 'Clicked', 'Returned']], dummy_vars], axis=1)

# Add constant for intercept
X = sm.add_constant(regression_data.drop('Dwell_Time', axis=1))
y = regression_data['Dwell_Time']

# Fit the model
model = sm.OLS(y, X).fit()
regression_summary = model.summary().tables[1]

# Output the regression results
print(model.summary())

# Create a box plot for visual comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Dwell_Time', data=ab_test_data)
plt.title('Comparison of Dwell Time Across Groups')
plt.xlabel('Group')
plt.ylabel('Dwell Time (seconds)')
plt.tight_layout()
plt.show()



