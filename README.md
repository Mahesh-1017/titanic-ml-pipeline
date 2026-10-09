# 🚢 Titanic Data Pipeline

<p align="center">
  <b>A modular, end-to-end data preprocessing pipeline built with Python, Pandas, Seaborn, and Scikit-learn.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

## 📌 Overview

The **Titanic Data Pipeline** is a Python-based data preprocessing project that transforms raw passenger data into a clean, structured, machine-learning-ready dataset.

Using the Titanic dataset, this project demonstrates essential data science practices, including data validation, text normalization, missing-value handling, categorical encoding, train-test splitting, and numerical feature scaling.

The pipeline is organized into reusable functions, making the preprocessing workflow easier to understand, maintain, test, and extend.

## 🎯 Project Objectives

- Validate the dataset schema and required columns.
- Clean and normalize categorical text.
- Remove unnecessary and redundant features.
- Handle missing numerical and categorical values.
- Encode categorical variables into numerical representations.
- Split the dataset into training and testing sets.
- Standardize numerical features without fitting the scaler on the test set.
- Prepare data for downstream machine learning models.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| Seaborn | Loading the Titanic dataset |
| Scikit-learn | Data splitting and feature scaling |
| Git & GitHub | Version control and project hosting |

## 📂 Project Structure

```text
Titanic-Data-Pipeline/
│
├── main.py              # Data preprocessing pipeline
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Files excluded from Git
```

## 📊 Dataset

**Source:** [Titanic Dataset — Seaborn](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)

The dataset contains passenger information such as:

- `survived` — survival outcome (target variable)
- `sex` — passenger sex
- `age` — passenger age
- `sibsp` — number of siblings or spouses aboard
- `parch` — number of parents or children aboard
- `fare` — passenger fare
- `embarked` — port of embarkation
- `class` — passenger ticket class

The dataset has 891 rows and 15 columns in the standard Seaborn version.

## ⚙️ Pipeline Workflow

```text
Raw Titanic Dataset
        │
        ▼
1. Load Dataset
        │
        ▼
2. Validate Required Columns
        │
        ▼
3. Clean Categorical Text
        │
        ▼
4. Remove Unnecessary Features
        │
        ▼
5. Handle Missing Values
        │
        ▼
6. Encode Categorical Variables
        │
        ▼
7. Split into Training and Testing Sets
        │
        ▼
8. Scale Numerical Features
        │
        ▼
Machine-Learning-Ready Data
```

### 1. Data Loading

Loads the Titanic dataset using Seaborn.

### 2. Data Validation

Checks whether all required columns are available and raises a custom `DataValidationError` when the schema is incomplete.

### 3. Text Cleaning

Removes leading and trailing whitespace and converts categorical text to lowercase.

### 4. Feature Selection

Removes unnecessary or redundant columns, including `alive`, `deck`, `adult_male`, `who`, `embark_town`, and `pclass`.

### 5. Missing-Value Handling

- Numerical features: median imputation.
- Categorical features: mode imputation.
- Target variable: preserved rather than imputed.

### 6. Categorical Encoding

- `sex`: male → 0, female → 1.
- `class`: first → 1, second → 2, third → 3.
- `embarked`: one-hot encoding.

### 7. Train-Test Split

Splits the data into:

- Training set: 80%
- Testing set: 20%

Uses `random_state=42` for reproducibility and stratifies the split using the survival target.

### 8. Feature Scaling

Uses `StandardScaler` to standardize the numerical features:

- `age`
- `sibsp`
- `parch`
- `fare`

The scaler is fitted on the training set and then applied to the testing set to prevent scaling-statistic leakage.

> **Implementation note:** In the current version, missing-value imputation occurs before the train-test split. For a fully leakage-safe workflow, imputation statistics should also be learned exclusively from training data.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- Git
- Visual Studio Code or another Python-compatible editor
- Internet connection to download the dataset through Seaborn

### 1. Clone the Repository

```bash
git clone https://github.com/Mahesh-1017/titanic-ml-pipeline.git
cd titanic-ml-pipeline
```

### 2. Create a Virtual Environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python main.py
```

## 📈 Expected Output

When execution succeeds, the terminal displays:

- Original dataset dimensions.
- Data validation confirmation.
- Cleaning and preprocessing status messages.
- Training and testing feature dimensions.
- Training and testing target dimensions.
- The first five rows of the processed training data.
- Missing-value counts for the training features.

The exact feature dimensions depend on the final preprocessing implementation.

## 🧠 Key Concepts Demonstrated

- Modular Python programming
- Data quality validation
- Exploratory data preparation
- Missing-value imputation
- Categorical feature engineering
- One-hot encoding
- Reproducible train-test splitting
- Feature standardization
- Prevention of data leakage
- Git-based version control

## 🔮 Future Improvements

- [ ] Add automated unit tests using `pytest`.
- [ ] Move imputation and encoding into Scikit-learn pipelines.
- [ ] Use `ColumnTransformer` for numerical and categorical preprocessing.
- [ ] Add exploratory data analysis and visualizations.
- [ ] Train Logistic Regression, Random Forest, and other classifiers.
- [ ] Evaluate models using precision, recall, F1-score, and ROC-AUC.
- [ ] Add a confusion matrix and feature-importance visualizations.
- [ ] Create a reproducible experiment and model comparison report.
- [ ] Add continuous integration with GitHub Actions.

## 👨‍💻 Author

**Mahesh Vinnakota**

- GitHub: [@Mahesh-1017](https://github.com/Mahesh-1017)
- LinkedIn: [Mahesh Vinnakota](https://www.linkedin.com/in/mahesh-vinnakota)

## 📄 License

This project is licensed under the MIT License. If you intend to distribute it under that license, add a `LICENSE` file to the repository.

---

<p align="center">
  <b>Built with Python, curiosity, and a passion for Machine Learning. 🚀</b>
</p>
