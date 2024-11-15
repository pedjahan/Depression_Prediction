import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Title is more than 31 characters. Some applications may not be able to read the file")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['data_path']
df = pd.read_csv(data_path)

# Feature Engineering
# --------------------

# Create 'Age Range Life Stage Based' feature
bins_life_age = [18, 25, 35, 45, 65, 150]
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
bins_health_age = [18, 40, 60, 150]
labels_health_age = ['Young (18-39)', 'Middle-Aged (40-59)', 'Older (60+)']
df['Age Range Health Based'] = pd.cut(
    df['Age'], bins=bins_health_age, labels=labels_health_age, right=False
)

# Create 'Income Category UK' feature
bins_income_uk = [0, 20000, 40000, 80000, 150000, float('inf')]
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

# ----------------------------------------

# create directories
os.makedirs('outputs/Insights', exist_ok=True)
# -------------------------

# EDA Functions
def plot_bar_with_count_percentage(df, columns):
    for column in columns:
        count = df[column].value_counts()
        categories = count.index
        total = count.sum()
        percentages = [(c / total) * 100 for c in count]
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=categories, y=count, palette='Paired')
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Counts')
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.annotate(f'{int(height)} ({percentages[i]:.1f}%)',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=6, fontsize=8)
        plt.yticks(fontsize=8)
        plot_filename = f"outputs/Insights/distribution_{column.replace(' ', '_')}.png"
        plt.savefig(plot_filename)
        plt.close()

# Columns to analyse
list_of_interesting_columns = [
    'History of Mental Illness', 
    'Marital Status',
    'Education Level', 
    'Number of Children',
    'Smoking Status', 
    'Physical Activity Level',
    'Employment Status', 
    'Alcohol Consumption',
    'Dietary Habits',
    'Sleep Patterns',
    'History of Substance Abuse', 
    'Family History of Depression',
    'Chronic Medical Conditions',
    'Age Range Life Stage Based',
    'Age Range Health Based', 
    'Income Category UK'
]

plot_bar_with_count_percentage(df, list_of_interesting_columns)

# Additional Function to Plot Distributions for Continuous Variables
def plot_continuous_distributions(df, continuous_columns):
    for column in continuous_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column].dropna(), kde=True, color="skyblue")  # kde adds a density plot
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plot_filename = f"outputs/Insights/distribution_{column.replace(' ', '_')}.png"
        plt.savefig(plot_filename)
        plt.close()

# Continuous columns to analyse
continuous_columns = ['Age', 'Income']  # Add other continuous columns as needed
plot_continuous_distributions(df, continuous_columns)

# Bivariate Analysis
def save_analysis_to_excel(results, file_name='bivariate_analysis.xlsx'):
    file_path = os.path.join("outputs/Insights", file_name)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        for (col, fixed_column), analysis_df in results.items():
            sheet_name = f'{col}_vs_{fixed_column}'
            analysis_df.to_excel(writer, sheet_name=sheet_name, index=False)

def bivariate_analysis_with_fixed_column(df, columns, fixed_column, tolerance=1.0):
    results = {}
    count = df[fixed_column].value_counts()
    total = count.sum()
    fixed_column_distribution_dict = (count / total * 100).to_dict()
    for col in columns:
        analysis_df = df.groupby([col, fixed_column]).size().reset_index(name='Count')
        analysis_df[f'Percentage by {col}'] = (
            analysis_df.groupby(col, group_keys=False)['Count'].apply(lambda x: (x / x.sum()) * 100).round(2)
        )
        total_count = analysis_df['Count'].sum()
        analysis_df['Overall Percentage'] = (analysis_df['Count'] / total_count * 100).round(2)
        def calculate_representation(row):
            category_percentage = fixed_column_distribution_dict.get(row[fixed_column], 0)
            percentage_by_col = row[f'Percentage by {col}']
            if percentage_by_col > category_percentage + tolerance:
                return "Overrepresented"
            elif percentage_by_col < category_percentage - tolerance:
                return "Underrepresented"
            else:
                return "Neutral"
        analysis_df['Representation Status'] = analysis_df.apply(calculate_representation, axis=1)
        results[(col, fixed_column)] = analysis_df
    return results

bivariate_columns = [
    'Marital Status',
    'Education Level',
    'Number of Children',
    'Smoking Status',
    'Physical Activity Level',
    'Employment Status',
    'Alcohol Consumption',
    'Dietary Habits',
    'Sleep Patterns',
    'History of Substance Abuse',
    'Family History of Depression',
    'Chronic Medical Conditions',
    'Age Range Life Stage Based',
    'Age Range Health Based',
    'Income Category UK'
]

bivariate_results = bivariate_analysis_with_fixed_column(df, bivariate_columns, 'History of Mental Illness')
save_analysis_to_excel(bivariate_results)