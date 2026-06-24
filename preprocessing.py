"""
preprocessing.py — تحميل البيانات، EDA، معالجة الـ Missing Values والـ Outliers،
                    Encoding، Train/Test Split، SMOTE، وScaling
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # لازم قبل import pyplot عشان Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (
    DATA_PATH, TARGET_COL, TEST_SIZE,
    RANDOM_STATE, OUTLIER_COLS
)


# ─────────────────────────────────────────────
# 1. تحميل البيانات
# ─────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ─────────────────────────────────────────────
# 2. EDA — بيرجعوا fig مش plt.show()
# ─────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    income_counts = df[TARGET_COL].value_counts()
    colors = ['#4CAF50', '#F44336']

    income_counts.plot(kind='bar', ax=axes[0], color=colors,
                       edgecolor='white', width=0.5)
    axes[0].set_title('Income Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of People')
    axes[0].tick_params(rotation=0)
    for bar in axes[0].patches:
        axes[0].annotate(
            f'{int(bar.get_height()):,}',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    axes[1].pie(income_counts, labels=income_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Percentage of Each Category', fontsize=14, fontweight='bold')

    plt.suptitle('🎯 Target Variable Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    palette = sns.color_palette('husl', len(num_cols))

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=40, color=palette[i],
                     edgecolor='white', alpha=0.85)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Frequency')

    plt.suptitle('📊 Distribution of Numeric Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_boxplots_by_income(df: pd.DataFrame):
    # educational-num (مش education-num) — ده الاسم الصح في adult.csv
    possible_cols = ['age', 'hours-per-week', 'educational-num', 'education-num', 'capital-gain']
    key_cols = [c for c in possible_cols if c in df.columns][:4]

    fig, axes = plt.subplots(1, len(key_cols), figsize=(16, 5))
    if len(key_cols) == 1:
        axes = [axes]

    for i, col in enumerate(key_cols):
        sns.boxplot(data=df, x=TARGET_COL, y=col, ax=axes[i],
                    palette=['#4CAF50', '#F44336'])
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Income')

    plt.suptitle('📦 Variable Distribution by Income', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. Missing Values & Duplicates
# ─────────────────────────────────────────────
def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dups = df.duplicated().sum()
    if dups > 0:
        df = df.drop_duplicates()
    return df


# ─────────────────────────────────────────────
# 4. Encoding
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_for_dummies = [c for c in categorical_cols if c != TARGET_COL]

    df = pd.get_dummies(df, columns=cat_for_dummies, drop_first=True)
    df[TARGET_COL] = df[TARGET_COL].map({'<=50K': 0, '>50K': 1})
    df = df.astype(int)
    return df


# ─────────────────────────────────────────────
# 5. Outlier Detection & Treatment
# ─────────────────────────────────────────────
def detect_outliers(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    outliers_report = {}

    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        count = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if count > 0:
            outliers_report[col] = count

    return outliers_report


def plot_outliers(outliers_report: dict):
    if not outliers_report:
        return None

    report_df = pd.DataFrame(
        list(outliers_report.items()), columns=['Column', 'Outlier Count']
    ).sort_values('Outlier Count', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(report_df) * 0.5)))
    bars = ax.barh(report_df['Column'], report_df['Outlier Count'],
                   color='#FF6B6B', edgecolor='white')
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width()):,}', va='center', fontsize=9)
    ax.set_title('🚨 Number of Outliers per Column', fontsize=14, fontweight='bold')
    ax.set_xlabel('Outlier Count')
    plt.tight_layout()
    return fig


def apply_log_transform(df: pd.DataFrame,
                        cols: list = ['capital-gain', 'capital-loss']) -> pd.DataFrame:
    """Log-transform skewed features (e.g. capital-gain, capital-loss)."""
    cols = [c for c in cols if c in df.columns]
    for col in cols:
        df[col] = np.log1p(df[col])
    return df


def treat_outliers(df: pd.DataFrame,
                   cols: list = OUTLIER_COLS) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


# ─────────────────────────────────────────────
# 6. Train/Test Split + SMOTE
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ─────────────────────────────────────────────
# 7. Scaling
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# Pipeline كامل
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(path: str = DATA_PATH):
    """
    Returns: X_train_scaled, X_test_scaled, y_train_resampled,
             y_test, scaler, feature_names, raw_df
    """
    df = load_data(path)
    raw_df = df.copy()   # نحتفظ بنسخة خام للـ EDA في Streamlit

    df = handle_missing_and_duplicates(df)
    df = apply_log_transform(df)          # ← Log Transform لـ capital-gain/loss
    df = encode_features(df)
    df = treat_outliers(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    feature_names = X_train.columns.tolist()
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_res, X_test)

    return X_train_scaled, X_test_scaled, y_train_res, y_test, scaler, feature_names, raw_df
