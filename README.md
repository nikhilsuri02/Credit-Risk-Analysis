# Credit Risk Analysis

Predicting whether a LendingClub loan will be paid off or charged off (defaulted) using the borrower's details at the time of application. Seven classification models are compared on a heavily imbalanced dataset. Built for the CSE343 Machine Learning course.

## The problem

A lender faces two kinds of risk. If it turns down a borrower who would have repaid, it loses business. If it approves a borrower who defaults, it loses money. Only about 20% of loans default, so a model can score well on accuracy just by predicting "paid off" every time. The useful question is how many real defaults a model catches, and at what cost in false alarms.

## Data

- **Source:** LendingClub accepted loans, 2007 to 2018 Q4 (`accepted_2007_to_2018Q4.csv.gz` from the Kaggle "lending-club" dataset). The file is too large for GitHub, so it is not included.
- **Target:** only loans with status `Fully Paid` (0) or `Charged Off` (1) are kept, about 80% and 20% of the total.

**Pre-processing**
- Dropped columns with more than 20% missing values.
- Kept only features available when the loan is issued, such as amount, term, interest rate, grade and sub-grade, employment length, home ownership, income and verification, purpose, state, debt-to-income ratio, credit history, revolving balance and utilisation, and FICO range.
- Added an average FICO score feature and one-hot encoded the categorical columns.
- Standardised the features and made a 75/25 train-test split.
- Undersampled the majority class in the training set to a 50/50 balance. The test set keeps the real 80/20 split.

<img src="Figures/Screenshots/17.png" alt="Class balance before and after undersampling" width="620">

## Models

Logistic Regression (with grid-search tuning), Random Forest, XGBoost, a neural network (Keras), Gaussian Naive Bayes, K-Nearest Neighbours and a linear SVM.

## Results

On the test set of 244,019 loans, where class 1 = default:

| Model | Defaults caught (recall) | Precision on defaults | F1 on defaults | Weighted F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.63 | 0.31 | 0.42 | 0.69 | 0.66 |
| Random Forest | 0.67 | 0.31 | 0.42 | 0.68 | 0.64 |
| XGBoost | 0.68 | 0.32 | 0.43 | 0.68 | 0.65 |
| Neural network | **0.72** | 0.30 | 0.43 | 0.66 | 0.62 |
| Gaussian Naive Bayes | 0.10 | 0.28 | 0.14 | 0.73 | 0.77 |
| K-Nearest Neighbours* | 0.07 | 0.53 | 0.12 | 0.74 | 0.80 |
| Linear SVM** | 0.35 | 0.39 | 0.37 | 0.77 | 0.77 |

\* Evaluated on a different split (305,023 loans).
\*\* Trained and tested on a 2,099-loan subsample, so it is not directly comparable.

**What this shows**
- Accuracy and weighted F1 are misleading here. KNN and Naive Bayes score highest on them, but they catch fewer than 1 in 10 defaults.
- Logistic Regression, Random Forest, XGBoost and the neural network all catch about two-thirds of defaults. In return, about 7 in 10 of the loans they flag would have been repaid.
- The ensembles and the neural network do no better than Logistic Regression, so the report picks Logistic Regression as the final model: same performance, simpler to explain.

| Logistic Regression confusion matrix | ROC curve |
|---|---|
| <img src="Figures/Screenshots/19.png" alt="Logistic Regression confusion matrix" width="380"> | <img src="Figures/Screenshots/15.png" alt="ROC curve, AUC 0.71" width="380"> |

## Exploratory analysis

Charts of the loan data, such as interest rate by grade, are in [`Figures/Screenshots`](Figures/Screenshots).

| Interest rate by grade | Loan term by interest rate range |
|---|---|
| <img src="Figures/Screenshots/13.png" alt="Interest rate distribution by loan grade" width="380"> | <img src="Figures/Screenshots/12.png" alt="Loan term and amount by interest rate range" width="380"> |

The full classification report for each model is in [`Figures`](Figures) (`1.PNG` to `7.PNG`).

## Repository layout

```
Code/
  ML_Final_Project.ipynb   final pipeline: Logistic Regression, Random Forest, XGBoost, neural network, Naive Bayes
  ML_Mid_Project.ipynb     midterm version: EDA and first models, including KNN and SVM
  Earlier/                 first draft
Figures/                   classification reports (1-7.PNG) and charts (Screenshots/)
PPTs + Report/             project report (PDF) and slides (PDF, PPTX)
```

## Running it

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost tensorflow seaborn matplotlib
```

Download `accepted_2007_to_2018Q4.csv.gz` from the Kaggle "lending-club" dataset, then update the `read_csv` path at the top of `Code/ML_Final_Project.ipynb` (it points at `/kaggle/input/...`).
