DATA_PATH = 'adult.csv'
TARGET_COL = 'income'

TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTLIER_COLS = ['capital-gain', 'capital-loss', 'fnlwgt', 'hours-per-week', 'age']

XGB_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.005,
    'max_depth': 8,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'scale_pos_weight': 1.1,
    'gamma': 0.3,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'aucpr',
}

MODEL_PATH = 'xgb_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'feature_columns.pkl'

DEFAULT_FEATURE_FLAGS = {
    'workclass_Private': 1,
    'education_HS-grad': 1,
    'occupation_Prof-specialty': 1,
    'relationship_Not-in-family': 1,
    'race_White': 1,
    'gender_Male': 1,
    'native-country_United-States': 1,
}