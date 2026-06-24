"""
main.py — لتدريب وحفظ الموديل من command line
شغّله مرة واحدة: python main.py
"""

from preprocessing import run_preprocessing_pipeline
from model import run_training_pipeline

def main():
    print("⏳ Running preprocessing pipeline...")
    (X_train_scaled, X_test_scaled,
     y_train, y_test,
     scaler, feature_names, _raw_df) = run_preprocessing_pipeline()

    print("🤖 Training model...")
    model, metrics = run_training_pipeline(
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        feature_names,
        scaler,
        save=True,
    )

    print(f"\n✅ Done!")
    print(f"   Accuracy : {metrics['accuracy']:.2%}")
    print(f"   AUC      : {metrics['roc_auc']:.4f}")
    print("   Artifacts saved: xgb_model.pkl, scaler.pkl, feature_columns.pkl")

if __name__ == '__main__':
    main()
