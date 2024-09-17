# Credit-Card-Fraud-Detection

This project implements machine learning models to detect fraudulent credit card transactions. It uses a dataset of transactions with labeled data indicating whether each transaction is fraudulent or not. The project includes data cleaning, feature encoding, and model training using Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.

# Features
Data Preprocessing: Cleaned the dataset by removing unnecessary columns and handling missing data.
Data Visualization: Visualized the distribution of fraudulent and non-fraudulent transactions.
Correlation Analysis: Created a heatmap to visualize the correlation between features.
Model Training: Trained multiple machine learning models, including Logistic Regression, Decision Tree, and Random Forest.
Model Evaluation: Evaluated the models using accuracy scores and plotted the results.
Prediction: Applied the trained models to predict fraudulent transactions in a test dataset.

# Dataset
The project uses the following datasets for training and testing:
fraudTrain.csv: The training dataset.
fraudTest.csv: The test dataset.
Both datasets include information about credit card transactions, such as transaction amount, merchant, and category, along with a label (is_fraud) indicating if the transaction was fraudulent.

# Requirements
Python 3.x
Libraries:
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

# Project Workflow
Data Loading:
Load the training and test datasets using Pandas.

Data Exploration:
Displayed basic information about the datasets, such as shape, column types, and summary statistics.
Checked for missing values and visualized the data distribution.

Data Cleaning:
Removed irrelevant columns like cc_num, first, last, street, city, state, zip, dob, trans_num, and trans_date_trans_time.
Encoded categorical features like merchant, category, gender, and job using label encoding.

Data Visualization:
Visualized the distribution of fraudulent vs. non-fraudulent transactions using a pie chart.
Created a heatmap to visualize feature correlations.

Model Training:
Trained three models: Logistic Regression, Random Forest Classifier, and Decision Tree Classifier.
Split the dataset into training and testing sets using an 80-20 split.

Model Evaluation:
Evaluated each model on the test dataset using the accuracy score.
Visualized the accuracy of each model using a line plot.

Prediction on Test Data:
Used the Random Forest Classifier to predict fraudulent transactions on the test dataset.
Calculated the accuracy of the predictions.

# Results
After evaluating the models, the results are presented in the form of a DataFrame and plotted for comparison. The accuracies of the three models are calculated and stored in the FinalResult DataFrame, which shows the performance of each model on the test data.
