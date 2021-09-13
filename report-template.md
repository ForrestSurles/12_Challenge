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

* ~~Describe the stages of the machine learning process you went through as part of this analysis.~~
* ~~Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).~~

The stages of the machine learning process are followed for both models: ***model/fit/predict/evaluate***.
 
  1. **Model**: an appropriate model is chosen for the data and instantiated. 
  
  2. **Fit**: (aka the training stage) the model learns how to adjust itself to make predictions that match the data being fed to it. This is where the distinction between the unmodified and resampled data is paramount.

  3. **Predict**: the model makes predictions for new data separate from the data it was fit to (or *trained on*)

  4. **Evaluate**: review performance of the model and determine its effectiveness using standard metrics
    - Example metrics include:
      - Accuracy
      - Precision
      - Recall (or *sensitivity*)
      - F1-Score

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* <font color=OrangeRed>Machine Learning Model 1</font>: Straightforward logistic regression model fit to ***unmodified*** dataset
  * ~~Description of Model 1 Accuracy, Precision, and Recall scores.~~

  - **Accuracy**: <font color=OrangeRed>95.20%</font>

    - The accuracy of the model measures ***how often*** the model was ***correct*** in its classification of high-risk loans

  $$accuracy = (TPs + TNs)\ ÷\ (TPs + TNs + FPs + FNs)$$

  - **Precision**: <font color=OrangeRed>85%</font>

    - The precision of the model measures the ***level of confidence*** in the model's ability to ***correctly*** make positive predictions

  $$ precision = TPs\ ÷\ (TPs + FPs)$$

  - **Recall**: <font color=OrangeRed>91%</font>

    - the recall of the model measures the number of ***actually high-risk*** loans the model ***correctly classified*** as high-risk

  $$ recall = TPs\ ÷\ (TPs + FNs) $$


* <font color=DeepSkyBlue>Machine Learning Model 2</font>: Logistic regression model fit to ***resampled*** dataset
  * ~~Description of Model 2 Accuracy, Precision, and Recall scores.~~

  - **Accuracy**: <font color=DeepSkyBlue>99.37%</font>

    - The accuracy of the model measures ***how often*** the model was ***correct*** in its classification of high-risk loans

  $$accuracy = (TPs + TNs)\ ÷\ (TPs + TNs + FPs + FNs)$$

  - **Precision**: <font color=DeepSkyBlue>84%</font>

    - The precision of the model measures the ***level of confidence*** in the model's ability to ***correctly*** make positive predictions

  $$ precision = TPs\ ÷\ (TPs + FPs)$$

  - **Recall**: <font color=DeepSkyBlue>99%</font>

    - the recall of the model measures the number of ***actually high-risk*** loans the model ***correctly classified*** as high-risk

  $$ recall = TPs\ ÷\ (TPs + FNs) $$

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* ~~Which one seems to perform best? How do you know it performs best?~~
* ~~Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )~~

Within the context of this credit risk analysis, it is more important to predict the `1`'s, *high-risk loans*, as opposed to the `0`'s, *low-risk loans*.

Based upon the comparison of the classification report generated for each model's performance using the `classification_report_imbalanced` function, <font color=DeepSkyBlue>Machine Learning Model 2</font> outperforms <font color=OrangeRed>Machine Learning Model 1</font>. This is best illustrated by the **F1-Score** of each model representing the model's *harmonic mean* or a single summary statistic encompassing the model's precision as well as its recall. <font color=DeepSkyBlue>Machine Learning Model 2</font> outperforms <font color=OrangeRed>Machine Learning Model 1</font> by a mere 3% which while not a large difference is nevertheless mathematically superior.

$$ F1\ =\ 2\ *\ (precision * recall)\ ÷\ (precision + recall) $$