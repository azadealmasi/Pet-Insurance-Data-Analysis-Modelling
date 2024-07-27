# Data Modelling- Undersampling
# In this script, we will train various machine learning models to predict whether a policy will lapse in the year. We will evaluate the models based on different performance metrics and identify the most important factors driving lapses.
# Import Necessary Libraries
# We import essential libraries for data manipulation, machine learning model building, evaluation, and visualization.

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# Load the Dataset
# We load the churn dataset from an Excel file and inspect the class distribution of the target variable.
# Load the Excel file
file_path = 'C:/Users/azade/OneDrive/Documents/ManyPets/Code/cleaned_data.xlsx'
xls = pd.ExcelFile(file_path)

# Load the churn dataset
churn_data = pd.read_excel(xls, sheet_name='churn_dataset')

# Drop the Policy_ID column
churn_data = churn_data.drop('Policy_ID', axis=1)

# Check class balance
class_counts = churn_data['Policy_Lapsed'].value_counts()

# Print class counts
print(class_counts)

plt.rcParams['axes.facecolor'] = '#F5F1EC'
plt.rcParams['axes.edgecolor'] = '#294638'
plt.rcParams['axes.labelcolor'] = '#294638'
plt.rcParams['xtick.color'] = '#294638'
plt.rcParams['ytick.color'] = '#294638'
plt.rcParams['text.color'] = '#294638'
pink = '#FF9D94'
light_green = '#44745E'

# Plot class distribution
plt.figure(figsize=(3, 2), dpi=250)
sns.barplot(x=class_counts.index, y=class_counts.values, palette=[pink, light_green], alpha=0.8)
plt.title('Class Distribution in the Dataset', fontsize=5.5)
plt.ylabel('Number of Occurrences', fontsize=5.5)
plt.xlabel('Policy Lapsed', fontsize=5.5)
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
plt.show()


# Feature Engineering
# We create new features to enhance the predictive power of our models.
# Create new features
churn_data['Policy_Duration'] = 2024 - churn_data['Policy_Start_Year']
churn_data['Owner_Age_Group'] = pd.cut(churn_data['Owner_Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
churn_data['Pet_Age_Group'] = pd.cut(churn_data['Pet_Age'], bins=[0, 2, 5, 10, 20], labels=['0-2', '3-5', '6-10', '11+'])
churn_data['Claims_per_Year'] = churn_data['Claim_History'] / churn_data['Policy_Duration']
churn_data['Vet_Visits_per_Year'] = churn_data['Veterinary_Visits'] / churn_data['Policy_Duration']
churn_data['Is_Long_Term_Customer'] = (churn_data['Policy_Duration'] > 5).astype(int)
churn_data['Policy_Cost_per_Pet_Age'] = churn_data['Policy_Annual_Cost'] / churn_data['Pet_Age']

# Display the first few rows of the updated dataset
print(churn_data.head())


# Prepare Data for Modeling
# We prepare the data by defining the target variable and features, and converting categorical columns into numerical ones using OneHotEncoder.
# Define the target variable and features
X = churn_data.drop('Policy_Lapsed', axis=1)
y = churn_data['Policy_Lapsed']

# Convert categorical columns to dummy variables using OneHotEncoder
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)


# Function to Clean Feature Names
# The `clean_feature_names` function processes feature names to remove prefixes and replace special characters, ensuring they are clean for visualization.

def clean_feature_names(feature_names):
    clean_names = []
    for name in feature_names:
        if name.startswith('remainder'):
            clean_name = name.replace('remainder__', '', 1)
        elif name.startswith('cat'):
            clean_name = name.replace('cat__', '', 1)
        else:
            clean_name = name
        clean_name = clean_name.replace('[', '_').replace(']', '_').replace('<', 'lt').replace('>', 'gt')
        clean_names.append(clean_name)
    return clean_names


# Function to Rank Features and Plot Cumulative Importance
# The `feature_ranking` function ranks the importance of features in a trained model using wrapper approach and plots their cumulative importance. It also identifies the most significant features based on a specified threshold.
# Function to rank features and plot cumulative importance
def feature_ranking(model, X_train_resampled, y_train_resampled, feature_names, model_name, fold_number, thr):
    model.fit(X_train_resampled, y_train_resampled)

    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame(model.feature_importances_, index=feature_names, columns=['importance']).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        feature_importances['cumulative_importance'] = feature_importances['importance'].cumsum() / feature_importances['importance'].sum()
        
        # Plot feature importances
        plt.figure(figsize=(3, 4), dpi=250)
        sns.barplot(x=feature_importances.importance, y=feature_importances.index, palette=[pink, light_green])
        plt.title(f'Feature Importances for {model_name} (Fold {fold_number})', fontsize=6)
        plt.xlabel('Importance', fontsize=6)
        plt.ylabel('Features', fontsize=6)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        plt.show()
        
        # Plot cumulative importance
        plt.figure(figsize=(3, 2), dpi=250)
        sns.lineplot(x=range(len(feature_importances)), y=feature_importances['cumulative_importance'], color=pink)
        plt.axhline(y=thr, color='r', linestyle='--')
        plt.title(f'Cumulative Importance of Features for {model_name} (Fold {fold_number})', fontsize=6)
        plt.xlabel('Number of Features', fontsize=6)
        plt.ylabel('Cumulative Importance', fontsize=6)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        plt.show()
        
        # Select features contributing to thr% of the importance
        selected_features = feature_importances[feature_importances['cumulative_importance'] <= thr].index
        return selected_features

    return feature_names


# Hyperparameter Tuning Function
# We define a function to perform hyperparameter tuning using either RandomizedSearchCV or GridSearchCV.

def hyperparameter_tuning(model, param_grid, X, y, search_type='random', n_iter=5):
    if search_type == 'random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
    else:
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    search.fit(X, y)
    print(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
    return search.best_estimator_


# Define Hyperparameters for Each Model
# We specify the hyperparameters to be tuned for each machine learning model.
# Define hyperparameters for each model
param_grids = {
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],

    },
    'XGBClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 6],
    },
    'LGBMClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 6],
        'num_leaves': [31, 40, 50]
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
}


# Cross-Validation and Model Evaluation
# 
# The `cross_validate_model` function performs k-fold cross-validation on a given machine learning model. It calculates and returns average performance metrics across all folds, while also generating ROC curves and confusion matrices for each fold.
# 
# Function Description
# 
# 1. **Initialization**:
#     - `StratifiedKFold` with 5 splits is used to maintain class balance across folds.
# 
# 2. **Cross-Validation Loop**:
#     - For each fold, the dataset is split into training and testing sets.
#     - Categorical features are transformed using `OneHotEncoder`.
#     - Feature names are cleaned for better readability.
#     - The training data is undersampled using `RandomUnderSampler` to handle class imbalance.
#     - Features are scaled using `StandardScaler`.
# 
# 3. **Feature Ranking and Selection**:
#     - LightGBM model is used to rank features based on importance.
#     - Cumulative importance is calculated, and features contributing to 99% of importance are selected.
# 
# 4. **Hyperparameter Tuning**:
#     - The model's hyperparameters are tuned using either `RandomizedSearchCV` or `GridSearchCV` with a specified parameter grid.
#     - The best model from hyperparameter tuning is used for further evaluation.
# 
# 5. **Model Training and Evaluation**:
#     - The best model is trained on the selected features from the training data.
#     - Predictions are made on the testing set.
#     - If the model supports probability predictions, ROC-AUC score and ROC curve are calculated.
#     - Various performance metrics (accuracy, balanced accuracy, precision, recall, F1 score, G-mean) are calculated and stored.
# 
# 6. **Results Aggregation**:
#     - Average performance metrics across all folds are calculated and printed.
#     - Mean ROC curve with standard deviation is plotted.
#     - Average confusion matrix is plotted.
# 
# 7. **Return**:
#     - The function returns a dictionary containing the average values of all performance metrics.

def cross_validate_model(model, model_name, X, y, thr=0.99):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'Accuracy': [], 'Balanced Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC-AUC': [], 'G-mean': []}
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    confusion_matrices = []

    for fold_number, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply OneHotEncoder to the training and testing data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Get feature names after transformation and clean them
        feature_names = clean_feature_names(preprocessor.get_feature_names_out())

        # Undersample the training data
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train_transformed, y_train)
        
        # Standard scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test_transformed)

        # Feature ranking and selection using LightGBM (for reference, not used for actual training)
        lgb_model = lgb.LGBMClassifier(random_state=42)
        selected_features = feature_ranking(lgb_model, X_train_scaled, y_train_resampled, feature_names, model_name, fold_number, thr)

        X_train_selected = pd.DataFrame(X_train_scaled, columns=feature_names).loc[:, selected_features]
        X_test_selected = pd.DataFrame(X_test_scaled, columns=feature_names).loc[:, selected_features]

        # Hyperparameter tuning
        best_model = hyperparameter_tuning(model, param_grids[model_name], X_train_selected, y_train_resampled)

        best_model.fit(X_train_selected, y_train_resampled)
        y_pred = best_model.predict(X_test_selected)

        if hasattr(best_model, "predict_proba"):
            y_pred_prob = best_model.predict_proba(X_test_selected)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            metrics['ROC-AUC'].append(roc_auc)

            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        # Append calculated metrics
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred))
        metrics['Recall'].append(recall_score(y_test, y_pred))
        metrics['F1 Score'].append(f1_score(y_test, y_pred))
        metrics['G-mean'].append(geometric_mean_score(y_test, y_pred))
        
        # Append confusion matrix
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # Print the average metrics
    print(f"\nModel: {model_name}")
    for metric in metrics:
        print(f"{metric}: {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")

    # Calculate mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(metrics['ROC-AUC'])
    std_roc_auc = np.std(metrics['ROC-AUC'])

    plt.figure(figsize=(4, 3), dpi=250)
    plt.plot(base_fpr, mean_tpr, color=pink, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc), lw=2, alpha=.8)
    plt.fill_between(base_fpr, np.maximum(mean_tpr - std_roc_auc, 0), np.minimum(mean_tpr + std_roc_auc, 1), color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color=light_green, alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=6)
    plt.ylabel('True Positive Rate', fontsize=6)
    plt.title(f'Receiver Operating Characteristic for {model_name}', fontsize=6)
    plt.legend(loc="lower right", fontsize=5)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.show()

    # Average confusion matrix
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    plt.rcParams['axes.facecolor'] = '#F5F1EC'
    plt.rcParams['axes.edgecolor'] = '#294638'
    plt.rcParams['axes.labelcolor'] = '#294638'
    plt.rcParams['xtick.color'] = '#294638'
    plt.rcParams['ytick.color'] = '#294638'
    plt.rcParams['text.color'] = '#294638'
    
    # Create a custom color map
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#ff9d94'])

    plt.figure(figsize=(5, 5), dpi=100)
    sns.heatmap(mean_confusion_matrix, annot=True, fmt='.2f', cmap=cmap, annot_kws={"size": 10, "color": "#294638"}, cbar_kws={'shrink': 0.75})
    plt.title(f'Confusion Matrix for {model_name}', fontsize=12, color='#294638')
    plt.xlabel('Predicted Label', fontsize=12, color='#294638')
    plt.ylabel('True Label', fontsize=12, color='#294638')
    plt.xticks(fontsize=10, color='#294638')
    plt.yticks(fontsize=10, color='#294638')
    plt.show()

    return {metric: np.mean(metrics[metric]) for metric in metrics}


# Model Training and Evaluation
# We train and evaluate several models including Random Forest, XGBoost, LightGBM, and Logistic Regression using the cross-validation function defined earlier.
# List to store results
results = []

# Random Forest with Hyperparameter Tuning 
rf_model = RandomForestClassifier(random_state=42)
results.append({'Model': 'RandomForestClassifier', **cross_validate_model(rf_model, "RandomForestClassifier", X, y)})

# XGBoost with Hyperparameter Tuning
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
results.append({'Model': 'XGBClassifier', **cross_validate_model(xgb_model, "XGBClassifier", X, y)})

# LightGBM with Hyperparameter Tuning
lgb_model = lgb.LGBMClassifier(random_state=42)
results.append({'Model': 'LGBMClassifier', **cross_validate_model(lgb_model, "LGBMClassifier", X, y)})

# Logistic Regression with Hyperparameter Tuning
lr_model = LogisticRegression(random_state=42)
results.append({'Model': 'LogisticRegression', **cross_validate_model(lr_model, "LogisticRegression", X, y)})

# Create a DataFrame to display results
results_df = pd.DataFrame(results)
print(results_df)

