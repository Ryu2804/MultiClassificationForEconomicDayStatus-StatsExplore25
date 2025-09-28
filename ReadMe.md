# Economic Day Status Prediction

## 🏆 Achievement
**Rank 1st in Public Score with 0.447021 RMSE**

## Prerequisites

### Software Requirements
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Git (for version control)

### Required Libraries
```bash
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install hdbscan>=0.8.27
```

### Alternative Installation
Install all dependencies at once:
```bash
pip install -r requirements.txt
```

### Hardware Recommendations
- **RAM**: Minimum 8GB, Recommended 16GB+ (for handling large datasets with missing values)
- **CPU**: Multi-core processor recommended (Random Forest utilizes parallel processing)
- **Storage**: At least 2GB free space for datasets and model outputs

### Knowledge Prerequisites
- **Python Programming**: Intermediate level
- **Machine Learning**: Understanding of classification algorithms, feature engineering
- **Data Science**: Experience with pandas, scikit-learn
- **Statistics**: Basic understanding of missing data patterns, cross-validation
- **Domain Knowledge**: Familiarity with economic indicators (helpful but not required)

### Dataset Requirements
- Access to the competition datasets (train.csv, test.csv)
- Preprocessed data from `00_DATA PREPARATION.ipynb`
- Sufficient disk space for intermediate files and model artifacts

## Overview

This project tackles a multi-class classification challenge to predict daily economic status across diverse global regions. The unique challenge involves handling datasets with **75% missing values (MCAR - Missing Completely At Random)** from 10 different regions spanning Europe, Asia, America, and Africa.

## 📊 Problem Statement

- **Task**: Multi-class classification for economic day status prediction
- **Target**: `economic_day_status` with classes: Low, Medium, High
- **Challenge**: 75% missing data across all features
- **Regions**: 10 diverse datasets representing global economic conditions
- **Evaluation Metric**: F1-macro score

## 🗂️ Project Structure

```
├── Data/
│   ├── train.csv                    # Combined and imputed training data
│   └── test.csv                     # Combined and imputed test data
├── 00_DATA PREPARATION.ipynb        # Data combination and imputation
├── 01_EXPLORATORY DATA ANALYSIS.ipynb  # Data exploration and insights
├── 02_FEATURE ENGINEERING AND MODELING.ipynb  # This notebook
├── 0.447021-submission.csv          # Final predictions
└── README.md
```

## 🔧 Methodology

### Data Preprocessing
- **Missing Data Handling**: Advanced imputation techniques for 75% MCAR data
- **Feature Engineering**: Strategic creation of economic indicators
- **Encoding**: One-hot encoding for categorical variables
- **Data Integration**: Combining 10 regional datasets

### Feature Engineering Highlights

#### 1. **Temporal Features**
```python
# Lagged features for trend analysis
inflation_rate_lag_1
unemployment_rate_lag_1
exchange_rate_lag_1
business_confidence_index_lag_1
```

#### 2. **Rolling Statistics**
```python
# Moving averages for smoothing
inflation_rate_ma_7    # 7-day moving average
unemployment_rate_ma_7
exchange_rate_std_7    # Volatility measure
```

#### 3. **Economic Ratios**
```python
# Key economic indicators
exports_imports_ratio = exports_usd / imports_usd
debt_gdp_x_interest_rate = debt_gdp_ratio * interest_rate
```

#### 4. **Rate of Change Features**
```python
# Momentum indicators
inflation_rate_pct_change
exchange_rate_daily_change
biz_confidence_change
```

#### 5. **Threshold-based Binary Features**
```python
# Economic condition flags
high_inflation = (inflation_rate > 0.05)
high_unemployment = (unemployment_rate > 0.06)
```

#### 6. **Categorical Encoding**
- Trade balance status: Deficit(-1), Neutral(0), Surplus(1)
- Political stability: Unstable(-1), Moderate(0), Stable(1)
- Income group: Low(-1), Lower-Middle(0), Upper-Middle(1)

### Model Architecture

**Random Forest Classifier** with optimized hyperparameters:
```python
RandomForestClassifier(
    n_estimators=805,
    max_depth=17,
    min_samples_split=14,
    min_samples_leaf=4,
    bootstrap=False,
    class_weight='balanced',
    random_state=42
)
```

### Key Features Used
- **Economic Indicators**: GDP per capita, inflation rate, unemployment rate
- **Trade Metrics**: Exports, imports, trade balance status
- **Financial Data**: Interest rates, exchange rates, debt ratios
- **Social Indicators**: Education index, healthcare index, poverty rate
- **Infrastructure**: Internet penetration, urbanization percentage
- **Governance**: Political stability, governance quality

## 📈 Results

- **Test F1-macro Score**: 0.4359
- **Public Leaderboard**: **Rank 1st with 0.447021 RMSE**
- **Model Performance**: Robust handling of imbalanced classes with `class_weight='balanced'`

## 🛠️ Technical Implementation

### Libraries Used
```python
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
```

### Key Functions
- `detailed_feature_engineering()`: Comprehensive feature creation
- `make_mi_scores()`: Mutual information analysis
- Model training with cross-validation
- Automated submission file generation

## 🎯 Success Factors

1. **Advanced Feature Engineering**: Created 20+ new features from existing data
2. **Missing Data Strategy**: Effective handling of 75% missing values
3. **Regional Diversity**: Successfully integrated data from 10 different regions
4. **Balanced Classification**: Addressed class imbalance with proper weighting
5. **Hyperparameter Optimization**: Fine-tuned model parameters for optimal performance

## 📋 Usage

1. **Data Preparation**: Run `00_DATA PREPARATION.ipynb` first
2. **Exploration**: Analyze data with `01_EXPLORATORY DATA ANALYSIS.ipynb`
3. **Modeling**: Execute `02_FEATURE ENGINEERING AND MODELING.ipynb`
4. **Submission**: Generated `0.447021-submission.csv` for competition

## 🔍 Feature Importance

The model identified key economic indicators through mutual information analysis, with features like:
- Economic ratios and trade balances
- Temporal patterns in economic metrics
- Regional and governance factors
- Infrastructure and social indicators

## 🏅 Competition Highlights

- Successfully handled one of the most challenging aspects of real-world data: extensive missing values
- Achieved top performance through strategic feature engineering rather than complex ensemble methods
- Demonstrated effective cross-regional economic modeling
- Robust performance across diverse economic conditions globally

---

*This project showcases advanced data science techniques for handling missing data and economic forecasting, achieving top-tier competition results through strategic feature engineering and robust modeling approaches.*