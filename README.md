# credit-risk-prediction
Machine Learning project predicting risk using the German Credit dataset and Python.
# Credit Risk Prediction Using Machine Learning

Project Overview

This project explores the use of machine learning to predict credit risk using the German Credit dataset from the UCI Machine Learning Repository.

The objective is to classify applicants as either good or bad credit risks and investigate how different machine learning models perform, with particular emphasis on identifying potentially high-risk borrowers.

Dataset

Source: UCI Machine Learning Repository — Statlog (German Credit Data), Dataset ID 144

The dataset contains 1,000 observations and 20 input features, including:

Credit amount and loan duration

Checking account and savings status

Credit history

Employment information

Age and housing status

The target variable contains two classes:

1: Good credit risk (70%)

2: Bad credit risk (30%)

Methodology

1. Data Preprocessing

Split the dataset into 80% training and 20% testing data using stratified sampling.

Applied One-Hot Encoding to categorical features.

Standardised numerical features using StandardScaler.

Used Scikit-learn pipelines to combine preprocessing and model training.

2. Model Training

Two classification algorithms were evaluated:

Logistic Regression: A linear classification model used as the baseline.

Random Forest: An ensemble learning algorithm used for comparison.

Logistic Regression was further optimised using GridSearchCV with five-fold cross-validation, selecting a regularisation parameter of C=0.1.

3. Model Evaluation

Models were evaluated using accuracy, precision, recall and F1-score.

Model

Accuracy

Bad-Credit Recall

Baseline Logistic Regression

74.0%

51.7%

Tuned Logistic Regression

75.5%

50.0%

Random Forest

76.0%

42.0%

Tuned Logistic Regression (0.35 threshold)

75.5%

73.3%

4. Threshold Optimisation

Because identifying bad-credit applicants is important in lending decisions, the classification threshold was adjusted to improve bad-credit recall.

Cross-validated training predictions were used to compare thresholds, with 0.35 achieving the highest bad-credit F1-score among the thresholds tested.

Selected Model Results

Model: Logistic Regression
Regularisation: C = 0.1
Classification threshold: 0.35

Metric

Result

Accuracy

75.5%

Bad-credit precision

57%

Bad-credit recall

73%

Bad-credit F1-score

64%

The selected model correctly identified 44 of the 60 bad-credit applicants in the test dataset.

This demonstrates how adjusting the decision threshold can improve the identification of risky applicants without substantially reducing overall accuracy.

Key Findings

Overall accuracy alone can be misleading when evaluating credit risk classifiers.

Random Forest achieved slightly higher accuracy but performed worse at identifying bad-credit applicants.

Adjusting the Logistic Regression classification threshold improved bad-credit recall from approximately 52% to 73%.

Model selection should consider the business consequences of false positives and false negatives.

Limitations

The dataset contains only 1,000 observations, which limits the reliability and generalisability of the results.

The test set was also consulted during model experimentation, meaning the reported test performance should be considered exploratory rather than a fully independent estimate of generalisation performance.

The model has not been evaluated for fairness, calibration or suitability for real-world lending decisions.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Jupyter Notebook

VS Code

Conclusion

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, classification, hyperparameter tuning, cross-validation, model comparison and threshold optimisation.

It highlights the importance of evaluating machine learning models against practical business objectives rather than relying exclusively on overall accuracy.