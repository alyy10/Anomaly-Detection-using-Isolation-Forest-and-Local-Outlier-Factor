# Anomaly-Detection-using-Isolation-Forest-and-Local-Outlier-Factor

This repository contains an end-to-end project demonstrating anomaly detection using two unsupervised machine learning algorithms: Isolation Forest and Local Outlier Factor (LOF). The project focuses on identifying fraudulent transactions within a credit card transactional dataset.

## Dataset

The dataset used in this project consists of credit card transactional data. It contains 15 numerical columns and 140,000 rows. The features are a result of a PCA transformation and are anonymized for privacy reasons, represented as C1, C2, ..., C12. The 'Class' column indicates whether a transaction is fraudulent (1) or legitimate (0).

## Project Structure

The repository is structured to provide a clear separation of concerns, with dedicated files for data preprocessing, model training, and utility functions.

### `engine.py`

This is the main script that orchestrates the entire anomaly detection pipeline. It performs the following key operations:

* **Configuration Loading**: Reads project configurations from `config.yaml`.
* **Data Loading**: Loads the credit card transactional data from the specified path.
* **Missing Value Imputation**: Handles null values in the dataset using median imputation, leveraging the `handle_null_values` function from `preprocessing.py`.
* **Contamination Calculation**: Determines the contamination score (proportion of anomalies) in the dataset, using `find_contamination` from `utils.py`.
* **Data Preparation**: Separates features (X) from the target variable (y).
* **Isolation Forest Model Training**: Trains an Isolation Forest model using the prepared data. The model is configured with 500 estimators and `max_samples` equal to the dataset size, with a contamination parameter derived from the dataset.
* **Isolation Forest Score Prediction**: Generates anomaly scores for the dataset using the trained Isolation Forest model.
* **Isolation Forest Model Saving**: Saves the trained Isolation Forest model using `pickle`.
* **Local Outlier Factor (LOF) Model Training**: Trains a Local Outlier Factor model with `n_neighbors=20` and the calculated contamination score.
* **LOF Anomaly Score Calculation**: Computes anomaly scores based on the LOF model.
* **LOF Model Saving**: Saves the trained LOF model using `pickle`.

This script demonstrates an end-to-end workflow for applying both Isolation Forest and Local Outlier Factor for anomaly detection.

### `model.py`

This file encapsulates the core machine learning model training and prediction logic. It provides functions for both Isolation Forest and Local Outlier Factor algorithms.

* **`train_IF(data)`**: Trains an Isolation Forest model. It takes the preprocessed data as input and returns a trained `IsolationForest` object. The `n_estimators` is set to 500, `max_samples` is dynamically set to the length of the input data, and `contamination` is a crucial parameter for anomaly detection.
* **`predict_scores(model, data)`**: Predicts anomaly scores using a trained Isolation Forest model. It returns the decision function scores, where lower scores indicate a higher likelihood of being an anomaly.
* **`train_lof(data)`**: Trains a Local Outlier Factor (LOF) model. This function takes the preprocessed data and returns a trained `LocalOutlierFactor` object. The `n_neighbors` parameter is set to 20, and `contamination` is also used here to define the proportion of outliers in the dataset.
* **`anomaly_scores(model)`**: Extracts the negative outlier factor from a trained LOF model. These scores represent the anomaly degree of each data point.
* **`save_model(model, framework, model_path)`**: Serializes and saves the trained machine learning models (either Isolation Forest or LOF) to a specified path using Python's `pickle` module. This allows for persistence and later reuse of the trained models without retraining.

### `preprocessing.py`

This file contains functions dedicated to data preprocessing steps, specifically handling missing values.

* **`handle_null_values(data)`**: This function takes a pandas DataFrame as input and imputes any missing (null) values with the median of their respective columns. This approach is chosen to minimize the impact of potential outliers on the imputation process, ensuring that the dataset is clean and ready for model training.

### `utils.py`

This file provides utility functions that support various aspects of the anomaly detection pipeline, including data reading, configuration loading, and contamination calculation.

* **`read_data_csv(file_path, **kwargs)`**: Reads a CSV file into a pandas DataFrame. This function is designed to be flexible, allowing additional keyword arguments to be passed directly to `pd.read_csv`.
* **`find_contamination(target_var, data)`**: Calculates the contamination score, which is the proportion of fraudulent transactions (anomalies) in the dataset. It takes the target variable column name and the DataFrame as input, and prints the count of fraudulent and normal transactions before returning the contamination ratio.
* **`read_config(path)`**: Reads a YAML configuration file. This function is crucial for loading parameters and paths defined in `config.yaml`, ensuring that the project is easily configurable and adaptable.

## Results

### Isolation Forest

* **Contamination**: The analysis revealed a contamination rate of approximately 0.0018, indicating that a very small percentage of transactions are fraudulent (253 fraudulent transactions out of 140,000 total transactions).
* **Anomaly Scores Distribution**: The Isolation Forest model generated decision function scores for each transaction. A histogram of these scores typically shows a distribution where lower scores correspond to anomalies. Transactions with scores below a certain threshold (e.g., 0.0019 as used in the notebook) were identified as anomalies.
* **Identified Anomalies**: The Isolation Forest model successfully identified transactions that it considered anomalous. These transactions often exhibited characteristics that deviated significantly from the majority of the data, as expected with an isolation-based approach.

### Local Outlier Factor (LOF)

* **Anomaly Scores**: The LOF model calculated a Negative Outlier Factor for each data point. Lower (more negative) LOF scores indicate a higher likelihood of being an outlier. The distribution of these scores helps in understanding the local density deviations.
* **Identified Outliers**: Similar to Isolation Forest, LOF successfully highlighted transactions that were locally isolated or had significantly different densities compared to their neighbors. This method is particularly effective in datasets where anomalies might be clustered or have varying densities across the feature space.

Both models effectively identified anomalies within the dataset, demonstrating their utility in unsupervised fraud detection. The choice between Isolation Forest and LOF often depends on the specific characteristics of the dataset and the nature of the anomalies being sought.

## Project Structure and Flow

This section outlines the overall architecture and data flow within the anomaly detection system. The process begins with data ingestion, followed by preprocessing, model training (for both Isolation Forest and Local Outlier Factor), and finally, anomaly detection and model saving.

<img width="454" height="898" alt="image" src="https://github.com/user-attachments/assets/64ca969a-62d3-48c6-9d92-35fb70728826" />

### Flow Description

1. **Data Ingestion**: The process starts with the `credit_card_transactional_data.csv` file, which serves as the primary data source.
2. **Data Reading**: The `utils.py` module, specifically the `read_data_csv` function, is responsible for loading this raw data into a pandas DataFrame.
3. **Main Orchestration**: The `engine.py` script acts as the central orchestrator, coordinating all subsequent steps. It also loads configuration parameters from `config.yaml`.
4. **Preprocessing**: The loaded data is then passed to `preprocessing.py`, where the `handle_null_values` function imputes missing values, ensuring data quality.
5. **Contamination Calculation**: After preprocessing, `utils.py`'s `find_contamination` function calculates the contamination score, a critical parameter for both anomaly detection models.
6. **Isolation Forest Pipeline**:
   * The preprocessed data and contamination score are used by `model.py` to train the `IsolationForest` model via `train_IF`.
   * Anomaly scores are then predicted using `predict_scores`.
   * Finally, the trained Isolation Forest model is saved to `IF_model.pkl` using `save_model`.
7. **Local Outlier Factor (LOF) Pipeline**:
   * Concurrently, the preprocessed data and contamination score are used by `model.py` to train the `LocalOutlierFactor` model via `train_lof`.
   * Anomaly scores are calculated using `anomaly_scores`.
   * The trained LOF model is then saved to `LOF_model.pkl` using `save_model`.

This modular design ensures maintainability, reusability, and clarity in the overall anomaly detection workflow.
