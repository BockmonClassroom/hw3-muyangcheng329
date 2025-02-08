import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#load datasets
active_mins = pd.read_csv("t1_user_active_min.csv")
variants = pd.read_csv("t2_user_variant.csv")
pre_experiment = pd.read_csv("t3_user_active_min_pre.csv")
user_attributes = pd.read_csv("t4_user_attributes.csv")

#merge
merged_data = active_mins.merge(variants[['uid', 'variant_number']], on='uid')

#calculate
grouped_stats = merged_data.groupby('variant_number')['active_mins'].agg(['mean', 'median'])

#T test
control = merged_data.loc[merged_data['variant_number'] == 0, 'active_mins']
treatment = merged_data.loc[merged_data['variant_number'] == 1, 'active_mins']
t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)

grouped_stats['t_statistic'] = t_stat
grouped_stats['p_value'] = p_value

#boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=merged_data['variant_number'], y=merged_data['active_mins'])
plt.xticks(ticks=[0, 1], labels=['Control (0)', 'Treatment (1)'])
plt.title("Active Minutes by Group")
plt.ylabel("Active Minutes")
plt.grid(True)
plt.show()

#clean outlier
data_no_outliers = merged_data[merged_data['active_mins'] <= 1440]

grouped_no_outliers = data_no_outliers.groupby('variant_number')['active_mins'].agg(['mean', 'median'])

#redo t-test
control_no_outliers = data_no_outliers.loc[data_no_outliers['variant_number'] == 0, 'active_mins']
treatment_no_outliers = data_no_outliers.loc[data_no_outliers['variant_number'] == 1, 'active_mins']
t_stat_new, p_value_new = stats.ttest_ind(control_no_outliers, treatment_no_outliers, equal_var=False)

grouped_no_outliers['t_statistic'] = t_stat_new
grouped_no_outliers['p_value'] = p_value_new

#merge

t3_avg = pre_experiment.groupby('uid')['active_mins'].mean().reset_index()
t3_avg.rename(columns={'active_mins': 'pre_experiment_mins'}, inplace=True)

merged_with_t3 = data_no_outliers.merge(t3_avg, on='uid', how='left')
merged_with_t3['activity_change'] = merged_with_t3['active_mins'] - merged_with_t3['pre_experiment_mins']

grouped_t3 = merged_with_t3.groupby('variant_number')[['active_mins', 'pre_experiment_mins', 'activity_change']].agg(['mean', 'median'])

#t-test
control_change = merged_with_t3.loc[merged_with_t3['variant_number'] == 0, 'activity_change']
treatment_change = merged_with_t3.loc[merged_with_t3['variant_number'] == 1, 'activity_change']
t_stat_t3, p_value_t3 = stats.ttest_ind(control_change, treatment_change, equal_var=False)

grouped_t3['t_statistic'] = t_stat_t3
grouped_t3['p_value'] = p_value_t3

#merge
total_data = merged_with_t3.merge(user_attributes, on='uid', how='left')

#calculate
grouped_user_type = total_data.groupby('user_type').agg(
    mean_active_mins=('active_mins', 'mean'),
    median_active_mins=('active_mins', 'median'),
    mean_pre_experiment_mins=('pre_experiment_mins', 'mean'),
    median_pre_experiment_mins=('pre_experiment_mins', 'median'),
    mean_activity_change=('activity_change', 'mean'),
    median_activity_change=('activity_change', 'median')
).reset_index()

#calculate
grouped_gender = total_data.groupby('gender').agg(
    mean_active_mins=('active_mins', 'mean'),
    median_active_mins=('active_mins', 'median'),
    mean_pre_experiment_mins=('pre_experiment_mins', 'mean'),
    median_pre_experiment_mins=('pre_experiment_mins', 'median'),
    mean_activity_change=('activity_change', 'mean'),
    median_activity_change=('activity_change', 'median')
).reset_index()

#print
print("T-Test Results:\n", grouped_stats)
print("T-Test Without Outliers:\n", grouped_no_outliers)
print("Pre-Experiment Analysis:\n", grouped_t3)
print("User Type Analysis:\n", grouped_user_type)
print("Gender Analysis:\n", grouped_gender)