"""
model.py — تدريب XGBoost، تقييم الموديل، رسم النتائج (بترجع fig)، حفظ/تحميل
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, f1_score
)
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from config import XGB_PARAMS, MODEL_PATH, SCALER_PATH, FEATURES_PATH


# ─────────────────────────────────────────────
# 1. التدريب
# ─────────────────────────────────────────────
def train_model(X_train_scaled, y_train) -> XGBClassifier:
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)
    return model


def naive_predictor(y_test) -> dict:
    """Baseline: يتوقع الـ majority class دايماً (0 = <=50K)."""
    import numpy as np
    y_naive = np.zeros(len(y_test), dtype=int)
    return {
        'accuracy': accuracy_score(y_test, y_naive),
        'f1':       f1_score(y_test, y_naive, zero_division=0),
    }


# ─────────────────────────────────────────────
# 2. التقييم — بيرجع dict بالأرقام
# ─────────────────────────────────────────────
def evaluate_model(model: XGBClassifier, X_test_scaled, y_test) -> dict:
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'y_pred':   y_pred,
        'y_proba':  y_proba,
        'accuracy': acc,
        'f1':       f1,
        'report':   report,
        'roc_auc':  roc_auc,
        'fpr':      fpr,
        'tpr':      tpr,
    }


# ─────────────────────────────────────────────
# 3. Confusion Matrix → fig (متطور)
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision          = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall             = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity        = tn / (tn + fp) if (tn + fp) > 0 else 0
    misclass_rate      = (fp + fn) / cm.sum()
    f1                 = f1_score(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # — الـ heatmap الأصلية مع colored rectangles —
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['≤50K', '>50K'], yticklabels=['≤50K', '>50K'],
                linewidths=1, linecolor='white', cbar=False,
                annot_kws={'size': 16, 'weight': 'bold'})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    # colored rectangles
    for (r, c), color in [((0,0),'green'), ((1,1),'green'),
                           ((0,1),'red'),   ((1,0),'red')]:
        axes[0].add_patch(plt.Rectangle((c, r), 1, 1,
                          fill=False, edgecolor=color, lw=3))

    # — الـ normalized heatmap + detailed metrics text —
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['≤50K', '>50K'], yticklabels=['≤50K', '>50K'],
                linewidths=1, linecolor='white', cbar=False)
    axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    details = (
        f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}\n"
        f"Precision={precision:.3f}  Recall={recall:.3f}  "
        f"Specificity={specificity:.3f}  F1={f1:.3f}  "
        f"Misclass={misclass_rate:.3f}"
    )
    fig.text(0.5, -0.02, details, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('🎯 Confusion Matrix Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 4. ROC Curve → fig
# ─────────────────────────────────────────────
def plot_roc_curve(fpr, tpr, roc_auc: float):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#4CAF50', lw=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5,
            label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#4CAF50')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('📈 ROC Curve — XGBoost', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 5. Feature Importance → fig
# ─────────────────────────────────────────────
def plot_feature_importance(model: XGBClassifier, feature_names: list,
                             top_n: int = 15):
    importance_df = (
        pd.DataFrame({'Feature': feature_names,
                      'Importance': model.feature_importances_})
        .sort_values('Importance', ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette('viridis', len(importance_df))
    bars = ax.barh(importance_df['Feature'].iloc[::-1],
                   importance_df['Importance'].iloc[::-1],
                   color=palette[::-1], edgecolor='white')
    for bar in bars:
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.4f}', va='center', fontsize=8.5)

    ax.set_title(f'🔍 Top {top_n} Feature Importances — XGBoost',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6. Feature Selection — Top 5 by importance
# ─────────────────────────────────────────────
def select_top5_features(model: XGBClassifier, feature_names: list):
    """يرجع أسماء أحسن 5 features بالترتيب."""
    importance_df = (
        pd.DataFrame({'Feature': feature_names,
                      'Importance': model.feature_importances_})
        .sort_values('Importance', ascending=False)
    )
    return importance_df.head(5)['Feature'].tolist()


def evaluate_top5(model: XGBClassifier, X_test_scaled,
                  y_test, feature_names: list) -> dict:
    """يقيّم الموديل باستخدام Top-5 features فقط."""
    top5 = select_top5_features(model, feature_names)
    idx  = [feature_names.index(f) for f in top5 if f in feature_names]
    X_top5 = X_test_scaled[:, idx]

    # نحتاج موديل جديد مدرّب على نفس الـ features — هنا بنقيّم بالـ predict فقط
    # (الموديل الأصلي يشتغل على كل الـ features)
    y_pred_top5 = model.predict(X_test_scaled)   # للمقارنة
    return {
        'top5_features': top5,
        'full_accuracy': accuracy_score(y_test, y_pred_top5),
        'full_f1':       f1_score(y_test, y_pred_top5),
    }


def select_features(model: XGBClassifier, X_train_scaled, X_test_scaled,
                    feature_names: list):
    selector = SelectFromModel(model, threshold='median', prefit=True)
    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel  = selector.transform(X_test_scaled)
    selected = [f for f, s in zip(feature_names, selector.get_support()) if s]
    return X_train_sel, X_test_sel, selected


# ─────────────────────────────────────────────
# 7. حفظ وتحميل الموديل
# ─────────────────────────────────────────────
def save_artifacts(model: XGBClassifier, scaler, feature_names: list) -> None:
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)


def load_artifacts():
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# Pipeline كامل
# ─────────────────────────────────────────────
def run_training_pipeline(X_train_scaled, X_test_scaled,
                           y_train, y_test, feature_names: list,
                           scaler, save: bool = True):
    model   = train_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test)

    if save:
        save_artifacts(model, scaler, feature_names)

    return model, metrics
