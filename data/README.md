# 📊 Data Directory

This directory contains the datasets used for the Credit Card Fraud Detection project.

## 📁 **Expected Files**

- `creditcard_2023.csv` - Main dataset (568,630 transactions)
- `sample_data.csv` - Sample dataset for testing
- Various processed samples created during pipeline execution

## 🔒 **Privacy Note**

The actual dataset files are excluded from Git due to:
- **Large file sizes** (325MB+ for main dataset)
- **Privacy considerations**
- **GitHub storage limitations**

## 📥 **To Use This Project**

1. **Download the dataset** from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or similar source
2. **Place the CSV file** in this directory as `creditcard_2023.csv`
3. **Run the pipeline** using `python main.py`

## 📋 **Dataset Requirements**

The dataset should have the following structure:
- **Features**: V1, V2, ..., V28 (PCA-transformed)
- **Amount**: Transaction amount
- **Class**: Target variable (0: Legitimate, 1: Fraudulent)

## 🔧 **Sample Data Generation**

The pipeline can generate sample datasets for testing:
```bash
# Create sample for quick testing
python main_fast.py
```

*Note: All large data files are automatically excluded from version control.*