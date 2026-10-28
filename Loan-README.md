# Loan Approval: Predicting Credit Risk with Python

**BAIM 3220 · Introduction to Python**

https://www.kaggle.com/competitions/3220-loan-approval

Every day, lenders decide who gets a loan and on what terms. Approve too few applicants and the bank loses business. Approve the wrong ones and it loses money. Either way, those decisions come from studying past borrowers: who they were, what they borrowed, and how things turned out.

In this project you'll use Python to answer one question:

> **Given what we know about a loan applicant, will the loan end up with `loan_status = 1`?**

You'll do this by entering our class Kaggle competition, [3220 Loan Approval](https://www.kaggle.com/competitions/3220-loan-approval). It's free and runs entirely in your browser.

You do **not** need any prior machine learning experience. If you can use pandas to load a CSV, filter rows, and run a `groupby`, you're ready.

> 📓 **Starter notebook:** `loan-starter.ipynb` follows this guide step by step. Fill in each `# TODO` cell as you read.

---

## Table of Contents

1. [What is machine learning?](#1-what-is-machine-learning)
2. [How a Kaggle competition works](#2-how-a-kaggle-competition-works)
3. [Getting set up](#3-getting-set-up)
4. [Meet the data](#4-meet-the-data)
5. [Step 1: Explore](#5-step-1-explore)
6. [Step 2: Your first prediction (no ML yet)](#6-step-2-your-first-prediction-no-ml-yet)
7. [Step 3: Prepare the data for a model](#7-step-3-prepare-the-data-for-a-model)
8. [Step 4: Fit a model](#8-step-4-fit-a-model)
9. [Step 5: Check your model honestly](#9-step-5-check-your-model-honestly)
10. [Step 6: Predict and submit](#10-step-6-predict-and-submit)
11. [Key vocabulary](#11-key-vocabulary)
12. [Ideas for improving your score](#12-ideas-for-improving-your-score)
13. [Common errors and fixes](#13-common-errors-and-fixes)
14. [What to turn in](#14-what-to-turn-in)

---

## 1. What is machine learning?

In regular programming, **you write the rules**:

```python
if loan_grade in ["D", "E", "F", "G"]:
    prediction = 1
else:
    prediction = 0
```

In machine learning, **the computer finds the rules for you** by studying examples where the answer is already known.

```
Regular programming:   data + rules     →  answers
Machine learning:      data + answers   →  rules (a "model")
```

Think of a new loan officer. Nobody hands them a perfect rulebook. Instead, they review thousands of past applications and their outcomes, and they start noticing patterns: high interest rates, large loans relative to income, a past default on file. Then they apply those patterns to new applications.

That's exactly what we'll do:

| Step | Everyday word | ML word |
|---|---|---|
| Look at past examples with known outcomes | Study | **Training** |
| Find patterns in those examples | Learn | **Fitting** the model |
| Apply patterns to new, unseen cases | Guess | **Predicting** |
| Check how good the guesses were | Grade | **Evaluating** |

This kind of problem, where we predict one of two categories (0 or 1), is called **binary classification**. It's the same type of problem as predicting customer churn, fraud, or spam.

---

## 2. How a Kaggle competition works

Kaggle gives everyone the same data, split into two files:

- **`train.csv`**: 58,645 loans **with** the answer (`loan_status` = 0 or 1). You learn from this.
- **`test.csv`**: 39,098 different loans **without** the answer. You predict these.

Kaggle secretly knows the answers for `test.csv`. When you upload your predictions, Kaggle compares them to the real answers and gives you a score: the **percentage you got right** (accuracy). Your score appears on the **leaderboard**.

```
train.csv ──► you build a model ──► model predicts test.csv ──► submission.csv ──► Kaggle scores it
```

### How you're scored

Kaggle scores this competition on **accuracy**: the percentage of loans where your 0/1 guess matches the real answer. That's the same as Titanic.

There's a catch. Only about **14%** of loans are 1s. So a "model" that predicts 0 for every loan is about **86% accurate** without learning anything. That means:

> ⚠️ **86% is the floor, not a good score.** Any model you build has to beat the score you'd get by always guessing the most common answer.

When one answer is much more common than the other, it's called **class imbalance**. You'll see it everywhere in business data: fraud, churn, and loan defaults are all rare.

> ⚠️ **Submit 0s and 1s.** `sample_submission.csv` shows `0.5` in every row. That's only a placeholder for the format. Your file should contain a `0` or a `1` for every loan.

---

## 3. Getting set up

1. **Create a free account** at [kaggle.com](https://www.kaggle.com) if you don't have one. You may use a nickname as your username.
2. Open the class competition: [3220 Loan Approval](https://www.kaggle.com/competitions/3220-loan-approval).
3. Click **Join Competition** and accept the rules.
4. Click the **Code** tab, then **New Notebook**. pandas, scikit-learn, matplotlib, and seaborn are already installed.
5. Upload `loan-starter.ipynb` (**File → Import Notebook**), or copy its cells in.
6. In the right-hand **Input** panel, confirm the competition data is attached. The starter notebook looks for it at `/kaggle/input/competitions/3220-loan-approval/`. If your Input panel shows a different path, use that one.

> 💡 **Prefer working on your laptop?** See `README.md` for local setup. Put `train.csv`, `test.csv`, and `sample_submission.csv` in the same folder as the notebook.

---

## 4. Meet the data

Each row is one loan application.

| Column | Meaning | Example |
|---|---|---|
| `id` | Unique ID number | `0` |
| `person_age` | Applicant's age in years | `37` |
| `person_income` | Annual income in dollars | `35000` |
| `person_home_ownership` | `RENT`, `OWN`, `MORTGAGE`, or `OTHER` | `RENT` |
| `person_emp_length` | Years employed | `0.0` |
| `loan_intent` | Purpose: `EDUCATION`, `MEDICAL`, `PERSONAL`, `VENTURE`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` | `EDUCATION` |
| `loan_grade` | Lender's risk grade, `A` (safest) to `G` (riskiest) | `B` |
| `loan_amnt` | Amount borrowed in dollars | `6000` |
| `loan_int_rate` | Interest rate (%) | `11.49` |
| `loan_percent_income` | Loan amount ÷ income | `0.17` |
| `cb_person_default_on_file` | Has the applicant defaulted before? `Y` / `N` | `N` |
| `cb_person_cred_hist_length` | Length of credit history in years | `14` |
| `loan_status` | **Target.** 0 or 1 (train only) | `0` |

**What does `loan_status = 1` mean?** The competition calls this "loan approval," but the pattern in the data looks like **credit risk**: riskier grades, higher interest rates, and past defaults all go with more 1s. As you explore, decide for yourself what a 1 seems to represent.

Two words you'll hear constantly:

- **Target** (or *label*): the thing we want to predict. Here, `loan_status`.
- **Features**: the columns we use to make the prediction, like `loan_grade` and `loan_int_rate`.

---

## 5. Step 1: Explore

Before building anything, get to know the data. This is the pandas you already know.

```python
import pandas as pd
from pathlib import Path

# Works on Kaggle and on your laptop
candidates = [Path("/kaggle/input/competitions/3220-loan-approval"), Path("."), Path("..")]
DATA_DIR = next(p for p in candidates if (p / "train.csv").exists())

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

train.head()
```

```python
train.info()          # column types and how many non-missing values
train.describe()      # summary statistics for numeric columns
train.isna().sum()    # count missing values in each column
```

Now ask questions of the data:

```python
# What fraction of loans are 1s overall?
train["loan_status"].mean()

# By the lender's risk grade?
train.groupby("loan_grade")["loan_status"].mean()

# By whether the applicant has defaulted before?
train.groupby("cb_person_default_on_file")["loan_status"].mean()

# By home ownership?
train.groupby("person_home_ownership")["loan_status"].mean()
```

You should see something striking. Grades A–C have rates around 5–14%, but at grade D the rate jumps to nearly 60%. Renters have far higher rates than homeowners.

> 🧠 **Think about it:** Why might grade D be such a sharp cutoff? Why might renters look riskier than people with mortgages? Does a high rate for a group prove that group *causes* the outcome?

---

## 6. Step 2: Your first prediction (no ML yet)

### 6a. The "always 0" baseline

First, measure the floor:

```python
(train["loan_status"] == 0).mean()
```

That's about **0.858**. Predicting 0 for everyone gets 85.8% right, so that's the very first score to beat.

### 6b. A one-rule baseline

Now use something we noticed: **predict 1 for grades D through G, and 0 otherwise.** Check it on the training data first:

```python
from sklearn.metrics import accuracy_score

rule = train["loan_grade"].isin(["D", "E", "F", "G"]).astype(int)
accuracy_score(train["loan_status"], rule)
```

That's about **0.879**, a real improvement over 0.858. Now build the submission from the test rows:

```python
predictions = test["loan_grade"].isin(["D", "E", "F", "G"]).astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "loan_status": predictions
})

submission.to_csv("submission.csv", index=False)
submission.head()
```

Submit it (see [Step 6](#10-step-6-predict-and-submit) for how). This is your **baseline**: the score your model needs to beat.

### 6c. Your turn: combine two rules

Loans that are more than 30% of the borrower's income are also very risky. Try predicting 1 when the grade is D–G **or** `loan_percent_income > 0.3`:

```python
rule2 = (train["loan_grade"].isin(["D", "E", "F", "G"]) | (train["loan_percent_income"] > 0.3)).astype(int)
accuracy_score(train["loan_status"], rule2)
```

That gets about **0.90**. Each rule you add by hand helps a little, but finding the best combination of dozens of rules by hand would take forever. That's the job of a model.

---

## 7. Step 3: Prepare the data for a model

Machine learning models only understand **numbers**, and they don't accept **missing values**.

### 7a. Check for missing values

```python
train.isna().sum().sum(), test.isna().sum().sum()
```

Good news: this dataset has **no missing values**. Real-world data almost never comes this clean, so enjoy it.

### 7b. Choose features

Start with the numeric columns plus two easy text columns:

```python
features = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
    "loan_grade", "cb_person_default_on_file",
]
```

We skip `person_home_ownership` and `loan_intent` for now. They need a different kind of conversion (see [Ideas for improving your score](#12-ideas-for-improving-your-score)).

### 7c. Turn text into numbers

`loan_grade` has a natural **order** (A is safest, G is riskiest), so map it to 1–7. `cb_person_default_on_file` is yes/no, so map it to 1/0.

```python
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

for df in [train, test]:
    df["loan_grade"] = df["loan_grade"].map(grade_map)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"N": 0, "Y": 1})
```

> ⚠️ **Use the same mapping for both files.** If you number the categories separately for `train` and `test`, a "B" could become 2 in one file and 3 in the other, and your model would quietly make nonsense predictions. Writing out the dictionary yourself guarantees the two files match.

### 7d. Separate features and target

By convention, we call the features `X` and the target `y`:

```python
X = train[features]
y = train["loan_status"]
X_test = test[features]
```

---

## 8. Step 4: Fit a model

We'll use a **decision tree**. It learns a flowchart of yes/no questions, like a game of 20 Questions:

```
             Is loan_percent_income > 0.31?
               /                      \
             No                       Yes
             |                         |
      Is loan_grade ≥ 4 (D)?     Is person_income < 39,000?
        /        \                  /          \
      ...        ...              ...          ...
```

The model decides **which questions to ask, and in what order**, by finding the splits that best separate the 1s from the 0s in the training data.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)
```

That one line, `model.fit(X, y)`, is the "learning."

What the settings mean:

- **`max_depth=4`**: the flowchart can ask at most 4 questions in a row. This keeps it from memorizing the data.
- **`random_state=42`**: makes your results repeatable.

Every scikit-learn model follows the same three-step pattern:

```python
model = SomeModel()          # 1. create
model.fit(X, y)              # 2. learn from training data
model.predict(X_new)         # 3. predict 0 or 1 for new data
```

---

## 9. Step 5: Check your model honestly

If we grade the model on the **same** loans it studied, it will look great, because it's already seen the answers.

### Overfitting

A model that memorizes the training data instead of learning general patterns is **overfit**. Try a tree with no depth limit:

```python
deep_model = DecisionTreeClassifier(random_state=42)  # no max_depth
deep_model.fit(X, y)
deep_model.score(X, y)   # accuracy on the training data
```

You'll get about **99.99%**. It got all but a handful of the 58,645 loans right, because it memorized them. Its real Kaggle score will be much lower.

### Hold out a validation set

The fix is to hide some training rows from the model, then test on them:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print("Training accuracy:  ", model.score(X_train, y_train))
print("Validation accuracy:", model.score(X_val, y_val))
```

- **Training accuracy** = how well it does on data it studied.
- **Validation accuracy** = how well it does on data it has never seen. **This is the number you should trust.**
- **`stratify=y`** keeps the same 14% share of 1s in both pieces.

A big gap between the two numbers is a warning sign of overfitting.

### Try it: find the best depth

```python
for depth in range(1, 16):
    m = DecisionTreeClassifier(max_depth=depth, random_state=42)
    m.fit(X_train, y_train)
    print(depth, round(m.score(X_train, y_train), 4), round(m.score(X_val, y_val), 4))
```

Training accuracy keeps climbing. Validation accuracy rises to about 0.915, levels off, then slowly **drops**. The sweet spot is where validation accuracy is highest.

> 💡 **Look at the fourth decimal place.** With 11,729 validation loans, a difference of 0.001 is about 12 loans. Depths that are within a few thousandths of each other are basically tied. When that happens, the shallower tree is usually the safer pick.

---

## 10. Step 6: Predict and submit

Once you've picked your settings, refit on **all** the training data (more examples means better learning), then predict the test loans:

```python
model = DecisionTreeClassifier(max_depth=8, random_state=42)   # use your best depth
model.fit(X, y)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "loan_status": predictions
})
submission.to_csv("submission.csv", index=False)
```

**To submit from a Kaggle notebook:**

1. Click **Save Version** (top right), then choose **Save & Run All** and wait for it to finish.
2. Open the saved version, go to the **Output** section, and find `submission.csv`.
3. Click **Submit**.
4. Check your score on the **Leaderboard** tab.

**To submit from your laptop:** on the competition page, click **Submit Prediction** and upload `submission.csv`.

Your submission file must have **exactly 39,098 rows** plus a header, and **exactly two columns**: `id` and `loan_status`. Every `loan_status` should be `0` or `1`.

---

## 11. Key vocabulary

| Term | Plain-English meaning |
|---|---|
| **Model** | A set of rules learned from data that turns inputs into a prediction |
| **Feature** | An input column used to make a prediction (e.g., `loan_int_rate`) |
| **Target / label** | The column we're trying to predict (`loan_status`) |
| **Training data** | Examples with known answers that the model learns from |
| **Test data** | New examples without answers that the model predicts |
| **Fitting** | The process of the model learning patterns from training data |
| **Classification** | Predicting a category (0 or 1) |
| **Accuracy** | Fraction of predictions that are correct |
| **Class imbalance** | When one answer is much more common than the other (here, 86% vs 14%) |
| **Baseline** | A simple starting score that any real model should beat |
| **Overfitting** | Memorizing training data instead of learning general patterns |
| **Validation set** | Training rows held back to check the model on unseen data |
| **Hyperparameter** | A setting you choose before fitting (e.g., `max_depth`) |
| **One-hot encoding** | Turning one text column into several 0/1 columns, one per category |

---

## 12. Ideas for improving your score

Try one change at a time and check your **validation accuracy** after each.

**Add the remaining text columns with one-hot encoding** (start here, it's the biggest win)

`person_home_ownership` and `loan_intent` have no natural order (is `MEDICAL` "bigger" than `VENTURE`?). Instead of numbering them, give each category its own 0/1 column:

```python
train = pd.get_dummies(train, columns=["person_home_ownership", "loan_intent"], dtype=int)
test  = pd.get_dummies(test,  columns=["person_home_ownership", "loan_intent"], dtype=int)
```

Then add the new column names (e.g., `person_home_ownership_RENT`) to your `features` list. Remember from Step 1 how differently renters and homeowners behaved.

**Create new features**

- `income_per_year_employed = person_income / (person_emp_length + 1)`
- `rate_x_percent_income = loan_int_rate * loan_percent_income`
- `age_minus_history = person_age - cb_person_cred_hist_length`

**Try other models** (same `fit` / `predict` pattern)

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
model = HistGradientBoostingClassifier(random_state=42)
```

A **random forest** builds many decision trees and lets them vote. **Gradient boosting** builds trees one after another, each fixing the previous one's mistakes. Both are often more accurate than a single tree.

> 🎯 **A realistic goal:** A decision tree on the starter features gets a validation accuracy of about 0.915. Adding the one-hot columns brings it to about 0.95, and random forests or gradient boosting land around 0.95 too. Scores above about 0.955 take real feature engineering and tuning.

---

## 13. Common errors and fixes

| Error message (or symptom) | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError` or `StopIteration` when loading | Data isn't where the notebook looks | Copy the exact path from Kaggle's **Input** panel, or put the CSVs next to the notebook |
| `ValueError: could not convert string to float: 'B'` | Text column not converted | Map `loan_grade` and `cb_person_default_on_file` to numbers, or one-hot encode |
| `ValueError: Input X contains NaN` | A `.map()` missed a category, creating blanks | Check `X.isna().sum()` and make sure your dictionary covers every category |
| `KeyError` on a one-hot column in `test` | A category exists in train but not test | Use `X_test = X_test.reindex(columns=X.columns, fill_value=0)` |
| Leaderboard score about 0.86 | Your model predicts 0 for (almost) every loan | Check `submission["loan_status"].value_counts()`. About 10–14% should be 1s |
| Submission error, or a very low score | `loan_status` holds decimals like `0.37` instead of 0/1 | Use `.predict()`, not `.predict_proba()` |
| Leaderboard score far below validation (about 0.75) | Rows in the wrong order, so predictions don't line up with `id` | Don't sort or filter `test` before building the submission |
| Submission rejected: wrong number of rows | Predicted on train instead of test, or filtered rows out | Predict on `X_test`. It must have 39,098 rows |
| Submission rejected: wrong columns | Extra index column, or wrong column names | Use `to_csv("submission.csv", index=False)` with columns `id` and `loan_status` |
| Validation accuracy much lower than training | Overfitting | Lower `max_depth`, or try a random forest |

---

## 14. What to turn in

1. Your **Kaggle username** (submitted through Canvas).
2. A **screenshot of your best leaderboard score**.
3. Your completed **`loan-starter.ipynb`** (shared Kaggle link or downloaded `.ipynb`), including:
   - your exploration, with at least three `groupby` questions and what you learned from each
   - your baseline submission and its score
   - your decision tree, its validation accuracy, and its Kaggle score
   - at least **one improvement attempt**, and whether it helped
   - a short paragraph, in your own words, explaining why **86% accuracy is not impressive** for this data
   - a short paragraph, in your own words, explaining what **overfitting** is and how you checked for it

---

### Further reading

- [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning), a free short course that builds on this project
- [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [Google ML Crash Course: Accuracy, precision, and recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), on why accuracy can mislead when one class is rare
