# Depression Data Analysis and Modelling

This project performs EDA, a simple feature engineering, and predictive modelling on a depression dataset. It includes visualisation of data distributions, bivariate analysis, and model training using Logistic Regression, Random Forest, and XGBoost.

## Table of Contents

- [Installation]
- [Usage]
- [Configuration]
- [Project Structure]
- [Results]
- [License]

## Installation

1. Clone the repository:
The project is unavailable online.

2. Install Dependencies and Required Libraries
It’s recommended to use a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

## Usage

1. Place your dataset in the data/ directory and update the config.yaml file with the dataset name and path.

2. Run the EDA script:
python src/eda.py

3. Run the data processing script:
python src/data_processing.py

4. Run the model training script:
python src/model_training.py

## Configuration

Edit the config.yaml file to set up your configuration:
	•	data_path: Path to your dataset.
	•	input_features: List of features to use as input.
	•	target_column: Name of the output feature column.
	•	algorithms: List of algorithms to use (e.g., ['RandomForest', 'LogisticRegression', 'XGBoost']).
	•	hyperparameters: Hyperparameters for each algorithm.
	•	random_grid_params: Parameters for RandomizedSearchCV including n_jobs, n_iter, and cv.

## Project Structure

• src/: Contains all the source code.
  • data_processing.py: Handles data loading and preprocessing.
  • eda.py: Contains exploratory data analysis code.
  • model_training.py: Handles model training and evaluation.
  • utils.py: Utility functions used across scripts.
• data/: Directory to store the dataset.
• outputs/: Contains generated outputs like plots, models, and performance metrics.
  • Insights/: Stores plots and analysis results.
  • Models/: Stores saved models.

## Results

After running the scripts, results including plots, trained models, and performance metrics will be available in the outputs/ directory.

## License
TBC