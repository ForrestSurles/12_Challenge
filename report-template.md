# Credit Risk Analysis Report

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* ~~Explain the purpose of the analysis.~~

The purpose of this analysis is to compare the capability of different supervised machine learning models, and determine whether, with such an inherently imbalanced dataset, a standard logistic regression model or an oversampled logistic regression model is more accurate in classifying the credit risk of a loan as *low-risk* or *high-risk*.

* ~~Explain what financial information the data was on, and what you needed to predict.~~

The dataset in question consists of common datapoints used to assess the creditworthiness of a borrower. The goal is to train a machine learning model to classify each loan in the dataset as a **high-risk loan** or a **low-risk loan** by utilizing and testing against existing classifications found in the `loan_status` column of the dataset. 

* ~~Provide basic information about the variables you were trying to predict (e.g., `value_counts`).~~

The primary reason for the comparison of these two techniques is due to the overwhelming imbalance inherent in this style of credit risk data. As a generalization, there will be vastly more *low-risk* loans than *high-risk* loans represented in the data. This fact can be observed by running the `value_counts` function on the labels (or *targets*) of each model.

For the unmodified model, `y.value_counts()` returns `75,036` *low-risk* loans and `2,500` *high-risk* loans. Since the number of *high-risk* loans only represents `~3.22%` of the entire dataset, this represents a very fundamental problem for the unmodified model because once it is fit to the `X_train` and `y_train` data output from the `train_test_split` function there is a concerningly high likelihood the model will have been ***overfit*** meaning the model will only properly classify *this particular dataset*, and if new data were to be introduced the model would not perform as expected.

For the resampled model, `y_resampled.value_counts()` returns `56,271` *low-risk* loans and `56,271` *high-risk* loans. Unlike the first model, the second model before being fit to the data had the data ***oversampled*** in order to mitigate the problem previously illustrated for the unmodified model. When the data is oversampled, this means more instances for the *smaller* class (i.e. *high-risk* loans) are generated in order to balance out the training the second model will undergo when fit to the resampled data.

* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).








## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
