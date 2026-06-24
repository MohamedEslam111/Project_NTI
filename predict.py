"""
predict.py — دالة التوقع على بيانات جديدة
"""

import pandas as pd
from config import DEFAULT_FEATURE_FLAGS


def predict_income(
    age: int,
    edu_num: int,
    hours: int,
    cap_gain: int,
    marital_status: str,
    model,
    scaler,
    feature_names: list,
) -> dict:
    """
    يتوقع مستوى الدخل بناءً على المدخلات.

    Returns dict:
        prediction  : 0 أو 1
        label       : '<=50K' أو '>50K'
        prob_high   : احتمالية >50K
        prob_low    : احتمالية <=50K
    """
    # 1. DataFrame فارغ بنفس أعمدة التدريب
    new_data = pd.DataFrame(0, index=[0], columns=feature_names)

    # 2. القيم المُدخلة من اليوزر
    for col_name, val in [
        ('age',            age),
        ('educational-num', edu_num),
        ('hours-per-week',  hours),
        ('capital-gain',    cap_gain),
    ]:
        if col_name in new_data.columns:
            new_data[col_name] = val

    # 3. الحالة الاجتماعية (One-Hot)
    marital_col = f'marital-status_{marital_status}'
    if marital_col in new_data.columns:
        new_data[marital_col] = 1
    else:
        # fallback
        for col in new_data.columns:
            if col.startswith('marital-status_Never'):
                new_data[col] = 1
                break

    # 4. باقي الـ defaults
    for col, val in DEFAULT_FEATURE_FLAGS.items():
        if col in new_data.columns:
            new_data[col] = val

    # 5. Scale → predict
    scaled   = scaler.transform(new_data)
    pred     = int(model.predict(scaled)[0])
    proba    = model.predict_proba(scaled)[0]

    return {
        'prediction': pred,
        'label':      '>50K' if pred == 1 else '<=50K',
        'prob_high':  float(proba[1]),
        'prob_low':   float(proba[0]),
    }
