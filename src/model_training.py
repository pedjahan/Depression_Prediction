import pandas as pd
import numpy as np
import yaml
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# data
train_df = pd.read_csv('outputs/train.csv')
val_df = pd.read_csv('outputs/val.csv')
test_df = pd.read_csv('outputs/test.csv')

# features and target
target_column = config['target_column']
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_val = val_df.drop(columns=[target_column])
y_val = val_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# initialise results dictionary
results = {}

# Random Grid Parameters
random_grid_params = config['random_grid_params']
n_jobs = random_grid_params['n_jobs']
n_iter = random_grid_params['n_iter']
cv = random_grid_params['cv']

# algorithms to use 
algorithms = config['algorithms']
hyperparameters = config['hyperparameters']

# Create directories for models and plots
os.makedirs('outputs/Models', exist_ok=True)
os.makedirs('outputs/Plots', exist_ok=True)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)',
                    ha='center', va='center', color='black', fontsize=10)
    
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'outputs/Plots/{model_name}_confusion_matrix.png')
    plt.close()

# Model Training
for algo in algorithms:
    if algo == 'LogisticRegression':
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        param_grid = hyperparameters['LogisticRegression']

        # Adjust param_grid for LogisticRegression to apply l1_ratio only with elasticnet
        if 'penalty' in param_grid and 'l1_ratio' in param_grid:
            penalties = param_grid['penalty']
            param_grid = [
                {'penalty': ['l1'], 'solver': param_grid['solver'], 'C': param_grid['C']},
                {'penalty': ['l2'], 'solver': param_grid['solver'], 'C': param_grid['C']},
                {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': param_grid['C'], 'l1_ratio': param_grid['l1_ratio']}
            ]
    elif algo == 'RandomForest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = hyperparameters['RandomForest']
    elif algo == 'XGBoost':
        model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        param_grid = hyperparameters['XGBoost']
    else:
        continue

    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        verbose=1,
        random_state=42,
        n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    results[algo] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=1),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    # Save the model
    model_filename = f'outputs/Models/best_{algo.lower()}_model.pkl'
    joblib.dump(best_model, model_filename)
    
    # create and save confusion matrix
    plot_confusion_matrix(y_test, y_pred, algo)

# saving performance metrics
performance_df = pd.DataFrame([
    {
        'Model': model,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'Best Params': metrics['best_params']
    }
    for model, metrics in results.items()
])
performance_df.to_csv('outputs/model_performance.csv', index=False)