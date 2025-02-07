import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro

#load datasets
t1 = pd.read_csv("t1_user_active_min.csv")
t2 = pd.read_csv("t2_user_variant.csv")
t3 = pd.read_csv("t3_user_active_min_pre.csv")
t4 = pd.read_csv("t4_user_attributes.csv")

#merge
merged_data = t1.merge(t2[['uid', 'variant_number']], on='uid', how='inner')

#calculate
grouped_stats = merged_data.groupby('variant_number')['active_mins'].agg(['mean', 'median'])

#T test
control_group = merged_data[merged_data['variant_number'] == 0]['active_mins']
treatment_group = merged_data[merged_data['variant_number'] == 1]['active_mins']
t_stat, p_value = stats.ttest_ind(control_group, treatment_group, equal_var=False)

#store
grouped_stats['t_statistic'] = t_stat
grouped_stats['p_value'] = p_value

#boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=merged_data['variant_number'], y=merged_data['active_mins'])
plt.xticks(ticks=[0, 1], labels=['Control (0)', 'Treatment (1)'])
plt.title("Boxplot of Active Minutes by Group")
plt.ylabel("Active Minutes")
plt.grid(True)
plt.show()

#clean outlier
cleaned_no_outliers = merged_data[merged_data['active_mins'] <= 1440]

#recalculate
grouped_no_outliers = cleaned_no_outliers.groupby('variant_number')['active_mins'].agg(['mean', 'median'])

#redo t-test
control_no_outliers = cleaned_no_outliers[cleaned_no_outliers['variant_number'] == 0]['active_mins']
treatment_no_outliers = cleaned_no_outliers[cleaned_no_outliers['variant_number'] == 1]['active_mins']
t_stat_new, p_value_new = stats.ttest_ind(control_no_outliers, treatment_no_outliers, equal_var=False)

#new t-test
grouped_no_outliers['t_statistic'] = t_stat_new
grouped_no_outliers['p_value'] = p_value_new

#merge
merged_with_t3 = cleaned_no_outliers.merge(
    t3.groupby('uid')['active_mins'].mean().reset_index().rename(columns={'active_mins': 'pre_experiment_mins'}),
    on='uid',
    how='left'
)

merged_with_t3['activity_change'] = merged_with_t3['active_mins'] - merged_with_t3['pre_experiment_mins']

#calculate
grouped_t3_analysis = merged_with_t3.groupby('variant_number')[['active_mins', 'pre_experiment_mins', 'activity_change']].agg(['mean', 'median'])

#t-test
control_change = merged_with_t3[merged_with_t3['variant_number'] == 0]['activity_change']
treatment_change = merged_with_t3[merged_with_t3['variant_number'] == 1]['activity_change']
t_stat_t3, p_value_t3 = stats.ttest_ind(control_change, treatment_change, equal_var=False)

grouped_t3_analysis['t_statistic'] = t_stat_t3
grouped_t3_analysis['p_value'] = p_value_t3

#merge
merged_with_t4 = merged_with_t3.merge(t4, on='uid', how='left')

#calculate
grouped_user_type = merged_with_t4.groupby('user_type').agg(
    mean_active_mins=('active_mins', 'mean'),
    median_active_mins=('active_mins', 'median'),
    mean_pre_experiment_mins=('pre_experiment_mins', 'mean'),
    median_pre_experiment_mins=('pre_experiment_mins', 'median'),
    mean_activity_change=('activity_change', 'mean'),
    median_activity_change=('activity_change', 'median')
).reset_index()

#calculate
grouped_gender = merged_with_t4.groupby('gender').agg(
    mean_active_mins=('active_mins', 'mean'),
    median_active_mins=('active_mins', 'median'),
    mean_pre_experiment_mins=('pre_experiment_mins', 'mean'),
    median_pre_experiment_mins=('pre_experiment_mins', 'median'),
    mean_activity_change=('activity_change', 'mean'),
    median_activity_change=('activity_change', 'median')
).reset_index()

#save
grouped_stats.to_csv("T_Test_Results.csv", index=False)
grouped_no_outliers.to_csv("T_Test_Without_Outliers.csv", index=False)
grouped_t3_analysis.to_csv("Pre_Experiment_Analysis.csv", index=False)
grouped_user_type.to_csv("User_Type_Analysis.csv", index=False)
grouped_gender.to_csv("Gender_Analysis.csv", index=False)

#print
print("T-Test Results:\n", grouped_stats)
print("T-Test Without Outliers:\n", grouped_no_outliers)
print("Pre-Experiment Analysis:\n", grouped_t3_analysis)
print("User Type Analysis:\n", grouped_user_type)
print("Gender Analysis:\n", grouped_gender)
