# Data Cleaning
# Overview
# In this script, we will preprocess and clean the provided datasets to prepare them for further analysis. We have four datasets containing information about our products, customers, and claims. We will perform the following steps:
# 1. Load the datasets.
# 2. Conduct basic exploratory data analysis (EDA).
# 3. Handle missing values.
# 4. Detect and handle outliers.
# 5. Convert data types.
# 6. Save the cleaned data to a new Excel file.
# 
# Importing Necessary Libraries
# First, let's import the necessary libraries for data manipulation, and preprocessing.
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# Loading the Datasets
# We will load the Excel file and extract the datasets from their respective sheets.

# Load the Excel file from the provided path
file_path = 'C:/Users/azade/OneDrive/Documents/ManyPets/Code/DataScienceTest_v1.5.xlsx'
xls = pd.ExcelFile(file_path)

# Load the sheets
policies_data_df = pd.read_excel(xls, sheet_name='policies_data')
claims_events_df = pd.read_excel(xls, sheet_name='claims_events')
claims_payments_df = pd.read_excel(xls, sheet_name='claims_payments')
churn_dataset_df = pd.read_excel(xls, sheet_name='churn_dataset')

# Display the first few rows of each dataframe to understand the structure and contents
(policies_data_df.head(), claims_events_df.head(), claims_payments_df.head(), churn_dataset_df.head())


# Basic Exploratory Data Analysis (EDA)
# Let's perform some basic EDA to understand the structure and content of each dataset. This includes examining the shape, data types, missing values, summary statistics, and first few rows of each dataset.

def numerical_summary(df):
   
    df = df.copy()
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].view('int64')  # Convert datetime to int64
    return df.describe()


def categorical_summary(df):
    
    categorical_features = df.select_dtypes(include=['object']).columns
    return df[categorical_features].describe()

# Applying the functions to the datasets
policies_numerical_summary = numerical_summary(policies_data_df)
policies_categorical_summary = categorical_summary(policies_data_df)

claims_events_numerical_summary = numerical_summary(claims_events_df)
claims_events_categorical_summary = categorical_summary(claims_events_df)

claims_payments_numerical_summary = numerical_summary(claims_payments_df)
claims_payments_categorical_summary = categorical_summary(claims_payments_df)

churn_dataset_numerical_summary = numerical_summary(churn_dataset_df)
churn_dataset_categorical_summary = categorical_summary(churn_dataset_df)


# Display the summaries
print("Policies Numerical Summary:")
print(policies_numerical_summary)
print("\nPolicies Categorical Summary:")
print(policies_categorical_summary)

print("\nClaims Events Numerical Summary:")
print(claims_events_numerical_summary)
print("\nClaims Events Categorical Summary:")
print(claims_events_categorical_summary)

print("\nClaims Payments Numerical Summary:")
print(claims_payments_numerical_summary)
print("\nClaims Payments Categorical Summary:")
print(claims_payments_categorical_summary)

print("\nChurn Dataset Numerical Summary:")
print(churn_dataset_numerical_summary)
print("\nChurn Dataset Categorical Summary:")
print(churn_dataset_categorical_summary)


# Handling Missing Values
# Next, we will detect and handle missing values in the datasets. We will use different strategies such as imputation for numerical columns and filling missing values with a new category for categorical columns.

def missing_values_ratio(df):
    return (df.isnull().sum() / len(df)) * 100

# Calculating the missing values ratio for each dataset
policies_missing_ratio = missing_values_ratio(policies_data_df)
claims_events_missing_ratio = missing_values_ratio(claims_events_df)
claims_payments_missing_ratio = missing_values_ratio(claims_payments_df)
churn_dataset_missing_ratio = missing_values_ratio(churn_dataset_df)

# Printing the missing values ratio for each dataset
print("Policies Data Missing Values Ratio (%):")
print(policies_missing_ratio)
print("\nClaims Events Data Missing Values Ratio (%):")
print(claims_events_missing_ratio)
print("\nClaims Payments Data Missing Values Ratio (%):")
print(claims_payments_missing_ratio)
print("\nChurn Dataset Missing Values Ratio (%):")
print(churn_dataset_missing_ratio)


# Imputing Missing Values
# We will handle missing values in the `amount_paid`using mean imputation.

# Impute missing values in 'amount_paid' with the mean
imputer = SimpleImputer(strategy='mean')
claims_payments_df['amount_paid'] = imputer.fit_transform(claims_payments_df[['amount_paid']])

# Verify the missing values in 'amount_paid' are fixed
claims_payments_missing_ratio_after_impute = missing_values_ratio(claims_payments_df)

print("Claims Payments Data Missing Values Ratio (%) After Imputation:")
print(claims_payments_missing_ratio_after_impute)


# Check the Type of Each Feature
# Convert all non-numerical columns in the datasets to string format

def convert_non_numerical_to_string(df):
    """
    Function to convert all non-numerical columns in a dataframe to string format.
    """
    non_numerical_columns = df.select_dtypes(exclude=['number']).columns
    df[non_numerical_columns] = df[non_numerical_columns].astype(str)
    return df

# Apply the conversion to all datasets
policies_data_df = convert_non_numerical_to_string(policies_data_df)
claims_events_df = convert_non_numerical_to_string(claims_events_df)
claims_payments_df = convert_non_numerical_to_string(claims_payments_df)
churn_dataset_df = convert_non_numerical_to_string(churn_dataset_df)

# Verify the changes
print("Policies Data Types:")
print(policies_data_df.dtypes)
print("\nClaims Events Data Types:")
print(claims_events_df.dtypes)
print("\nClaims Payments Data Types:")
print(claims_payments_df.dtypes)
print("\nChurn Dataset Data Types:")
print(churn_dataset_df.dtypes)


# Manual Data Cleaning
# At first glance, we observed negative values for pet_age_at_purchase. We will remove the rows in the policies_data that inclue negative age.

policies_data_df = policies_data_df[policies_data_df['pet_age_at_purchase'] >= 0]

# Verify the removal of negative ages
negative_age_count = (policies_data_df['pet_age_at_purchase'] < 0).sum()
policies_data_summary_after_negatives_removal = policies_data_df['pet_age_at_purchase'].describe()
print("Count of negative ages after removal:", negative_age_count)
print("\nSummary statistics for pet_age_at_purchase after removing negative ages:")
print(policies_data_summary_after_negatives_removal)


# Detecting Outliers
# In this section, we define a function `handling_outliers` that can detect outliers in a specified column of a dataframe using the IQR method. We then use this function to calculate the outlier ratios for each numerical column in our datasets. 

def handling_outliers(df, method, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if method == 'detection':
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_ratio = len(outliers) / len(df) * 100
        return outlier_ratio
    elif method == 'removing':
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df_filtered
    
def calculate_outlier_ratios(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    outlier_ratios = {col: handling_outliers(df, 'detection', col) for col in numerical_columns}
    return outlier_ratios
        
policies_outlier_ratios = calculate_outlier_ratios(policies_data_df)
claims_events_outlier_ratios = calculate_outlier_ratios(claims_events_df)
claims_payments_outlier_ratios = calculate_outlier_ratios(claims_payments_df)
churn_dataset_outlier_ratios = calculate_outlier_ratios(churn_dataset_df)

# Print the outlier ratios for each dataset
print("Policies Data Outlier Ratios (%):")
print(policies_outlier_ratios)
print("\nClaims Events Data Outlier Ratios (%):")
print(claims_events_outlier_ratios)
print("\nClaims Payments Data Outlier Ratios (%):")
print(claims_payments_outlier_ratios)
print("\nChurn Dataset Outlier Ratios (%):")
print(churn_dataset_outlier_ratios)

policies_data_df = policies_data_df[policies_data_df['pet_age_at_purchase'] >= 0]


# Removing Outliers
# In this section, we use the same `handling_outliers` function to remove outliers. Since we found outliers in the `premium` and `pet_age_at_purchase` columns during the detection phase, we will focus on removing outliers from these two columns in the policies dataset. After removing the outliers, we verify the effectiveness of the removal by recalculating and printing the outlier ratios for the cleaned data.

columns_to_clean = ['premium', 'pet_age_at_purchase']

for column in columns_to_clean:
    policies_data_df = handling_outliers(policies_data_df, 'removing', column)

policies_outlier_ratios_after_removal = calculate_outlier_ratios(policies_data_df)

print("Policies Data Outlier Ratios After Removal (%):")
print(policies_outlier_ratios_after_removal)

print(policies_data_df['premium'].describe())
print(policies_data_df['pet_age_at_purchase'].describe())


# The revised summary indicates a more representative distribution of premium and pet_age_at_purchase values without the influence of extreme outliers.

# Now we double check the data quality to verify that the code is cleaned
# Display the summaries after cleaning
print("Policies Numerical Summary:")
print(numerical_summary(policies_data_df))
print("\nPolicies Categorical Summary:")
print(categorical_summary(policies_data_df))

print("\nClaims Events Numerical Summary:")
print(numerical_summary(claims_events_df))
print("\nClaims Events Categorical Summary:")
print(categorical_summary(claims_events_df))

print("\nClaims Payments Numerical Summary:")
print(numerical_summary(claims_payments_df))
print("\nClaims Payments Categorical Summary:")
print(categorical_summary(claims_payments_df))

print("\nChurn Dataset Numerical Summary:")
print(numerical_summary(churn_dataset_df))
print("\nChurn Dataset Categorical Summary:")
print(categorical_summary(churn_dataset_df))


# Saving the Cleaned Data
# Finally, we save the cleaned dataframes to a new Excel file with multiple sheets for further analysis.
# Save the final version of each dataframe to a single Excel file with multiple sheets
with pd.ExcelWriter('C:/Users/azade/OneDrive/Documents/ManyPets/Code2/cleaned_data.xlsx') as writer:
    policies_data_df.to_excel(writer, sheet_name='policies_data', index=False)
    claims_events_df.to_excel(writer, sheet_name='claims_events', index=False)
    claims_payments_df.to_excel(writer, sheet_name='claims_payments', index=False)
    churn_dataset_df.to_excel(writer, sheet_name='churn_dataset', index=False)

print("Dataframes saved to a single Excel file with multiple sheets successfully.")




