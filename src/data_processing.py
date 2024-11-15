import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load data
data_path = config['data_path']
df = pd.read_csv(data_path)

# -----------------------------------

# Feature engineering

# Create 'Age Range Life Stage Based' feature
bins_life_age = [18, 25, 35, 45, 65, df['Age'].max() + 1]
labels_life_age = [
    'Early Adulthood (18-24)',
    'Young Professionals (25-34)',
    'Established Adults (35-44)',
    'Mature Adults (45-64)',
    'Retirement (65+)'
]
df['Age Range Life Stage Based'] = pd.cut(
    df['Age'], bins=bins_life_age, labels=labels_life_age, right=False
)

# Create 'Age Range Health Based' feature
bins_health_age = [18, 40, 60, df['Age'].max() + 1]
labels_health_age = ['Young (18-39)', 'Middle-Aged (40-59)', 'Older (60+)']
df['Age Range Health Based'] = pd.cut(
    df['Age'], bins=bins_health_age, labels=labels_health_age, right=False
)

# Create 'Income Category UK' feature
bins_income_uk = [0, 20000, 40000, 80000, 150000, df['Income'].max() + 1]
labels_income_uk = [
    'Low Income (<£20k)',
    'Middle Income (£20k-£40k)',
    'Upper-Middle Income (£40k-£80k)',
    'High Income (£80k-£150k)',
    'Very High Income (£150k+)'
]
df['Income Category UK'] = pd.cut(
    df['Income'], bins=bins_income_uk, labels=labels_income_uk, right=False
)

# ----------------------

# Select input features and target
input_features = config['input_features']
target_column = config['target_column']
df = df[input_features + [target_column]]

# -----------

# Preprocessing
# Handle categorical variables

# Exclude the target column from one-hot encoding
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
if target_column in categorical_columns:
    categorical_columns.remove(target_column)

# Perform one-hot encoding on all categorical columns except the target column
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Encode target column if it's categorical
if df[target_column].dtype == 'object' or str(df[target_column].dtype) == 'category':
    df[target_column] = df[target_column].astype('category').cat.codes

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Outlier Capping
def cap_outliers(df, column, lower_bound, upper_bound):
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

# numerical columns for scaling and outlier handling
numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude the target column from numerical columns
if target_column in numerical_columns:
    numerical_columns = [col for col in numerical_columns if col != target_column]

# Ensure numerical_columns is not empty
if len(numerical_columns) == 0:
    print("No numerical columns found for scaling.")
else:
    # Proceed with outlier capping and scaling
    outlier_bounds = {}
    for col in numerical_columns:
        lower_bound = train_df[col].quantile(0.05)
        upper_bound = train_df[col].quantile(0.95)
        outlier_bounds[col] = (lower_bound, upper_bound)
        cap_outliers(train_df, col, lower_bound, upper_bound)

    for col, (lower, upper) in outlier_bounds.items():
        cap_outliers(val_df, col, lower, upper)
        cap_outliers(test_df, col, lower, upper)

    # Scaling
    scaler = StandardScaler()
    train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
    val_df[numerical_columns] = scaler.transform(val_df[numerical_columns])
    test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

    # Save scaler
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(scaler, 'outputs/scaler.pkl')

# Save processed data
os.makedirs('outputs', exist_ok=True)
train_df.to_csv('outputs/train.csv', index=False)
val_df.to_csv('outputs/val.csv', index=False)
test_df.to_csv('outputs/test.csv', index=False)