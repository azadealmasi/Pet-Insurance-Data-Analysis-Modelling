# Data Analysis
# ## Overview
# In this task, we address three business questions using datasets containing information about our products, customers, and claims. The provided datasets are used to answer the following questions:
# 
# 1. How many policies are in the dataset? How many of them are for cats? How many policies had pets older than 6 years when the policy was purchased?
# 2. What is the average loss ratio per species and breed?
# 3. What is the average time to make a claim for cats and dogs? Are these times significantly different?

# ## Importing Necessary Libraries
# First, we need to import the necessary libraries for data manipulation and statistical analysis.

import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import shapiro, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer


# 1. Policy Analysis
# We will load the provided datasets and answer the following:
# 
# * Total number of policies
# * Number of policies for cats
# * Number of policies for dogs
# * Number of policies where the pet was older than 6 years at the time of purchase

# Loading the Data
# We start by loading the Excel file and its sheets into dataframes.

file_path = 'C:/Users/azade/OneDrive/Documents/ManyPets/Code/cleaned_data.xlsx'
xls = pd.ExcelFile(file_path)

policies_data_cleaned = pd.read_excel(xls, sheet_name='policies_data')
claims_events_cleaned = pd.read_excel(xls, sheet_name='claims_events')
claims_payments_cleaned = pd.read_excel(xls, sheet_name='claims_payments')
churn_dataset_cleaned = pd.read_excel(xls, sheet_name='churn_dataset')


# Calculating Required Statistics
# We calculate the total number of policies, the number of policies for cats, dogs, and the number of policies with pets older than 6 years at the time of purchase.

total_policies = policies_data_cleaned['policy_id'].nunique()
cat_policies = policies_data_cleaned[policies_data_cleaned['species'] == 'Cat']['policy_id'].nunique()
dog_policies = policies_data_cleaned[policies_data_cleaned['species'] == 'Dog']['policy_id'].nunique()
older_pets_policies = policies_data_cleaned[policies_data_cleaned['pet_age_at_purchase'] > 6]['policy_id'].nunique()

print(f"Total number of policies: {total_policies}")
print(f"Number of cat policies: {cat_policies}")
print(f"Number of dog policies: {dog_policies}")
print(f"Number of policies for pets older than 6 at purchase: {older_pets_policies}")


# 2.1 Average Loss Ratio per Species
# The loss ratio is defined as the amount paid in loss divided by the amount collected in premium. We calculate the average loss ratio per species using three datasets: `policies_data`, `claims_events`, and `claims_payments`.

# Merging Datasets
# We merge the necessary datasets to compute the loss ratio.
merged_claims = pd.merge(policies_data_cleaned, claims_events_cleaned, on='policy_id')
complete_data = pd.merge(merged_claims, claims_payments_cleaned, on='claim_id')


# Overview the complete_data for calculating loss ratio
complete_data_premium_stats = complete_data['premium'].describe()
complete_data_premium_stats

complete_data_premium_stats = complete_data['amount_paid'].describe()
complete_data_premium_stats


# Checking For Normality
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = '#F5F1EC'
plt.rcParams['axes.edgecolor'] = '#294638'
plt.rcParams['axes.labelcolor'] = '#294638'
plt.rcParams['xtick.color'] = '#294638'
plt.rcParams['ytick.color'] = '#294638'
plt.rcParams['text.color'] = '#294638'
pink = '#FF9D94'
# Plotting histograms for premium and amount_paid columns
plt.figure(figsize=(12, 5))

# Histogram for premium
plt.subplot(1, 2, 1)
sns.histplot(complete_data['premium'], color = pink ,kde=True)
plt.title('Histogram of Premium')

# Histogram for amount_paid
plt.subplot(1, 2, 2)
sns.histplot(complete_data['amount_paid'], color = pink, kde=True)
plt.title('Histogram of Amount Paid')

plt.tight_layout()
plt.show()

# Conducting Shapiro-Wilk test
shapiro_test_premium = shapiro(complete_data['premium'])
shapiro_test_amount_paid = shapiro(complete_data['amount_paid'])

shapiro_test_premium, shapiro_test_amount_paid


# The Shapiro-Wilk test is used to check the normality of a distribution. The null hypothesis (H0) for this test is that the data is normally distributed.
# According to the histogram and the result of shapiro test, the p-value for premium is pvalue=2.6776131924335808e-30, and for amount_paid is is 1.5029849093540965e-20, We reject the null hypothesis, suggesting that there is  significant deviation from normality.
# Therefore we need to nomalize the data.

# Normalization
# Yeo-Johnson Transformation (can handle zero and negative values)
pt = PowerTransformer(method='yeo-johnson')
complete_data['premium_normalized'] = pt.fit_transform(complete_data['premium'].values.reshape(-1, 1))
complete_data['amount_paid_normalized'] = pt.fit_transform(complete_data['amount_paid'].values.reshape(-1, 1))

# Shapiro-Wilk Test on Yeo-Johnson Transformed Data
shapiro_test_premium = shapiro(complete_data['premium_normalized'])
shapiro_test_amount = shapiro(complete_data['amount_paid_normalized'])
print(shapiro_test_premium)
print(shapiro_test_amount)


import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = '#F5F1EC'
plt.rcParams['axes.edgecolor'] = '#294638'
plt.rcParams['axes.labelcolor'] = '#294638'
plt.rcParams['xtick.color'] = '#294638'
plt.rcParams['ytick.color'] = '#294638'
plt.rcParams['text.color'] = '#294638'

# Plotting histograms for premium_normalized and amount_paid_normalized columns
plt.figure(figsize=(12, 5))

# Histogram for premium
plt.subplot(1, 2, 1)
sns.histplot(complete_data['premium_normalized'],color = pink, kde=True)
plt.title('Histogram of premium_normalized')

# Histogram for amount_paid
plt.subplot(1, 2, 2)
sns.histplot(complete_data['amount_paid_normalized'], color = pink, kde=True)
plt.title('Histogram of amount_paid_normalized')

plt.tight_layout()
plt.show()


# The Yeo-Johnson transformation provided a significant improvement in normality for premium. For amount_paid, the improvement was notable but not perfect.

# Calculating Loss Ratio per species
# We calculate the loss ratio and then the average loss ratio per species.
average_loss_ratio = complete_data.groupby('species').agg(
    total_premium_collected=('premium_normalized', 'sum'),
    total_amount_paid=('amount_paid_normalized', 'sum')
).reset_index()

average_loss_ratio['average_loss_ratio'] = average_loss_ratio['total_amount_paid'] / average_loss_ratio['total_premium_collected']

print(average_loss_ratio)


# 2.2 Average Loss Ratio per Breed
# The following code calculates the average loss ratio per breed category using four datasets: `policies_data`, `claims_events`, `claims_payments`, and `churn_dataset`.

# Merging Datasets
# * The `churn_dataset` is preprocessed by adding a '0' at the beginning of each `Policy_ID` and converting the column to a string format.
# * The `churn_dataset` is merged with the previously merged dataset on the `policy_id` column.

churn_dataset_cleaned['policy_id'] = '0' + churn_dataset_cleaned['Policy_ID'].astype(str)
merged_data_breed = pd.merge(complete_data, churn_dataset_cleaned, on='policy_id')


# Overview the merged_data_breed for calculating loss ratio
merged_data_breed_premium_stats = merged_data_breed['premium'].describe()
merged_data_breed_premium_stats

merged_data_breed_premium_stats = merged_data_breed['amount_paid'].describe()
merged_data_breed_premium_stats


# Checking merged_data for normality
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = '#F5F1EC'
plt.rcParams['axes.edgecolor'] = '#294638'
plt.rcParams['axes.labelcolor'] = '#294638'
plt.rcParams['xtick.color'] = '#294638'
plt.rcParams['ytick.color'] = '#294638'
plt.rcParams['text.color'] = '#294638'
pink = '#FF9D94'

# Plotting histograms for premium and amount_paid columns
plt.figure(figsize=(12, 5))

# Histogram for premium
plt.subplot(1, 2, 1)
sns.histplot(merged_data_breed['premium'], color = pink, kde=True)
plt.title('Histogram of Premium')

# Histogram for amount_paid
plt.subplot(1, 2, 2)
sns.histplot(merged_data_breed['amount_paid'], color = pink, kde=True)
plt.title('Histogram of Amount Paid')

plt.tight_layout()
plt.show()

# Conducting Shapiro-Wilk test
shapiro_test_premium = shapiro(merged_data_breed['premium'])
shapiro_test_amount_paid = shapiro(merged_data_breed['amount_paid'])

shapiro_test_premium, shapiro_test_amount_paid


# According to the Shapiro-Wilk test We fail to reject the null hypothesis, suggesting that there is no significant deviation from normality for the premium, and amount paid data.

# Calculating loss ratio per breed
# The average loss ratio is computed for each breed category by grouping the data by the `Breed_Category` column and calculating the mean of the `loss_ratio`.
average_loss_ratio_breed = merged_data_breed.groupby('Breed_Category').agg(
    total_premium_collected_breed=('premium', 'sum'),
    total_amount_paid_breed=('amount_paid', 'sum')
).reset_index()

average_loss_ratio_breed['average_loss_ratio_breed'] = average_loss_ratio_breed ['total_amount_paid_breed'] / average_loss_ratio_breed ['total_premium_collected_breed']

print(average_loss_ratio_breed )


# 3. Time to Claim Analysis
# We measure the time to claim as the number of days between the policy inception date and the first claim. We calculate the average time to make a claim for cats and dogs and test if these times are significantly different.

# Converting the date columns to datetime format
claims_events_cleaned['claim_received'] = pd.to_datetime(claims_events_cleaned['claim_received'])
claims_events_cleaned['policy_inception_date'] = pd.to_datetime(claims_events_cleaned['policy_inception_date'])


# Identifying First Claim Dates
# We identify the first claim date for each policy ID.
first_claim_dates = claims_events_cleaned.groupby('policy_id')['claim_received'].min().reset_index()
merged_df = pd.merge(first_claim_dates, claims_events_cleaned[['policy_id', 'policy_inception_date', 'pet_type']], on='policy_id')
merged_df = merged_df.drop_duplicates(subset=['policy_id'])


# Calculating AVG Time to Claim
# We calculate the time to claim in days.
merged_df['AVG_first_time_to_claim'] = (merged_df['claim_received'] - merged_df['policy_inception_date']).dt.days
average_time_to_claim = merged_df.groupby('pet_type')['AVG_first_time_to_claim'].mean().reset_index()

print(average_time_to_claim)


# Statistical Testing
# We perform normality tests and appropriate statistical tests to determine if the time to claim is significantly different between cats and dogs.
cats_first_time_to_claim = merged_df[merged_df['pet_type'] == 'Cat']['AVG_first_time_to_claim']
dogs_first_time_to_claim = merged_df[merged_df['pet_type'] == 'Dog']['AVG_first_time_to_claim']

cat_normality_test = shapiro(cats_first_time_to_claim)
dog_normality_test = shapiro(dogs_first_time_to_claim)

print(f"Shapiro-Wilk test for cats: W={cat_normality_test.statistic}, p-value={cat_normality_test.pvalue}")
print(f"Shapiro-Wilk test for dogs: W={dog_normality_test.statistic}, p-value={dog_normality_test.pvalue}")

if cat_normality_test.pvalue > 0.05 and dog_normality_test.pvalue > 0.05:
    t_test = ttest_ind(cats_first_time_to_claim, dogs_first_time_to_claim)
    print(f"T-test: t-statistic={t_test.statistic}, p-value={t_test.pvalue}")
else:
    mann_whitney_test = mannwhitneyu(cats_first_time_to_claim, dogs_first_time_to_claim)
    print(f"Mann-Whitney U test: U-statistic={mann_whitney_test.statistic}, p-value={mann_whitney_test.pvalue}")


# This analysis provides insights into the number of policies, the loss ratios, and the claim times for different species, helping to inform business decisions based on data-driven insights.
