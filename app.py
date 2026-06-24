"""
app.py — Streamlit dashboard لمشروع Income Prediction
شغّله بـ: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, DATA_PATH
from preprocessing import (
    run_preprocessing_pipeline,
    plot_target_distribution,
    plot_numeric_distributions,
    plot_boxplots_by_income,
    plot_outliers,
    detect_outliers,
    load_data,
    encode_features,
    handle_missing_and_duplicates,
)
from model import (
    run_training_pipeline,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    load_artifacts,
    naive_predictor,
    select_top5_features,
)
from predict import predict_income


# ════════════════════════════════════════════════════════════════════════════
# CSS Styling
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 24px;
    color: white;
}
.hero-banner h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 12px 0 8px 0;
    color: white;
}
.hero-banner p {
    font-size: 1rem;
    opacity: 0.9;
    margin: 0;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    color: white;
    margin-bottom: 8px;
}
.metric-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}

.about-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.about-card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0 0 12px 0;
    color: #1a1a2e;
}
.about-card p {
    color: #555;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

.section-header {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.section-header h2 {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
    color: #1a1a2e;
}

.tags-container {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-top: 16px;
}
.tags-container h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 14px 0;
    color: #1a1a2e;
}
.tag {
    display: inline-block;
    background: #f0f4ff;
    color: #667eea;
    border: 1px solid #d6e0ff;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 4px 4px 4px 0;
}

.cm-box {
    background: white;
    border-radius: 12px;
    padding: 16px 24px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
    margin-bottom: 16px;
}

.dataset-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-top: 16px;
}
.dataset-card h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 12px 0;
    color: #1a1a2e;
}
.dataset-card ul {
    margin: 0;
    padding-left: 16px;
    color: #555;
    font-size: 0.88rem;
    line-height: 2;
}

.result-high {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: white;
    margin: 16px 0;
}
.result-low {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: white;
    margin: 16px 0;
}
.result-high h2, .result-low h2 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    color: white;
}

.footer {
    text-align: center;
    color: #888;
    font-size: 0.8rem;
    padding: 32px 0 16px 0;
    border-top: 1px solid #f0f0f0;
    margin-top: 32px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Cached helpers
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# Cached helpers
# ════════════════════════════════════════════════════════════════════════════

# 1. دالة لمراقبة أي تغيير في ملف الإعدادات
def get_config_timestamp():
    return os.path.getmtime('config.py')

# 2. إضافة التايم ستامب للدالة عشان الكاش يفرغ لوحده لما تعدل الـ config
@st.cache_resource(show_spinner="🤖 Training model with updated parameters...")
def get_all_artifacts(config_stamp):
    # بمجرد ما config_stamp يتغير، الكاش هيعتبر الدالة "جديدة" وهينفذها بالكامل
    
    # تنفيذ الـ Preprocessing Pipeline
    (X_train_scaled, X_test_scaled,
     y_train, y_test,
     scaler, feature_names, raw_df) = run_preprocessing_pipeline()

    # تشغيل التدريب بالبارامترات المحدثة (دائماً نسيف التعديلات الجديدة)
    model, metrics = run_training_pipeline(
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        feature_names, scaler, save=True,
    )
    
    return model, scaler, feature_names, metrics, raw_df, y_test, metrics['y_pred']

# 3. تمرير الـ timestamp للدالة
model, scaler, feature_names, metrics, raw_df, y_test, y_pred = get_all_artifacts(get_config_timestamp())

# Derived values (باقي كودك زي ما هو بالظبط بدون تغيير)
top5_features = select_top5_features(model, feature_names)
feat_display = {
    'capital-gain': 'Capital Gain',
    'marital-status_ Married-civ-spouse': 'Marital Status (Married)',
    'marital-status_Married-civ-spouse': 'Marital Status (Married)',
    'age': 'Age',
    'education-num': 'Education Level',
    'educational-num': 'Education Level',
    'hours-per-week': 'Hours per Week',
    'relationship_ Husband': 'Relationship (Husband)',
    'relationship_Husband': 'Relationship (Husband)',
    'capital-loss': 'Capital Loss',
    'occupation_ Prof-specialty': 'Occupation (Prof)',
}
top5_display = [feat_display.get(f, f) for f in top5_features]

r      = metrics['report']
acc    = metrics['accuracy']
f1     = metrics.get('f1', r['1']['f1-score'])
prec   = r['1']['precision']
recall = r['1']['recall']
naive  = naive_predictor(y_test)

# ════════════════════════════════════════════════════════════════════════════
# Navigation
# ════════════════════════════════════════════════════════════════════════════
page = st.radio(
    "nav",
    ["🔮 Predict", "📊 EDA", "🤖 Model Performance", "🏠 Overview"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Predict
# ════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":

    st.markdown("""
    <div class="hero-banner">
        <div style="font-size:3rem">💰</div>
        <h1>Income Prediction App</h1>
        <p>Predict whether an individual earns more than $50K using machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header"><h2>👤 Enter Individual Information</h2></div>',
                    unsafe_allow_html=True)

        with st.form("predict_form"):
            age = st.number_input("Age", min_value=17, max_value=90, value=35)

            workclass = st.selectbox("Work Class", [
                'Private', 'Self-emp-not-inc', 'Self-emp-inc',
                'Federal-gov', 'Local-gov', 'State-gov',
                'Without-pay', 'Never-worked'
            ])

            education = st.selectbox("Education Level", [
                'Bachelors', 'Some-college', '11th', 'HS-grad',
                'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                '7th-8th', '12th', 'Masters', '1st-4th', '10th',
                'Doctorate', '5th-6th', 'Preschool'
            ])

            edu_num = st.number_input("Education Years", min_value=1, max_value=16, value=10)

            marital = st.selectbox("Marital Status", [
                'Married-civ-spouse', 'Never-married', 'Divorced',
                'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
            ])

            occupation = st.selectbox("Occupation", [
                'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
            ])

            relationship = st.selectbox("Relationship", [
                'Wife', 'Own-child', 'Husband', 'Not-in-family',
                'Other-relative', 'Unmarried'
            ])

            race = st.selectbox("Race", [
                'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
            ])

            sex = st.selectbox("Sex", ['Male', 'Female'])

            cap_gain = st.number_input("Capital Gain", min_value=0, max_value=10000000, value=0)
            cap_loss = st.number_input("Capital Loss", min_value=0, max_value=10000000, value=0)

            hours = st.slider("Hours per Week", 1, 99, 40)

            country = st.selectbox("Native Country", [
                'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
                'Japan', 'Greece', 'England', 'China', 'Germany', 'Iran',
                'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
                'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
                'Vietnam', 'Honduras', 'Hungary', 'Scotland', 'Yugoslavia',
                'Peru', 'Hong', 'Ireland', 'Canada', 'Other'
            ])

            submitted = st.form_submit_button(
                "🌍 Predict Income Level",
                use_container_width=True,
                type="primary"
            )

    with col_right:
        # Model Performance
        st.markdown('<div class="section-header"><h2>📊 Model Performance</h2></div>',
                    unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">ACCURACY</div>
                <div class="value">{acc:.1%}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">RECALL</div>
                <div class="value">{recall:.0%}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">PRECISION</div>
                <div class="value">{prec:.0%}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">F1 SCORE</div>
                <div class="value">{f1:.2f}</div>
            </div>""", unsafe_allow_html=True)

        # CM checkmark/cross visual
        st.markdown("""
        <div class="cm-box">
            <span style="font-size:2.2rem;color:#22c55e">✓</span>
            &nbsp;&nbsp;&nbsp;
            <span style="font-size:2.2rem;color:#ef4444">✗</span>
            <div style="color:#888;font-size:0.75rem;margin-top:8px;">
                Correct vs Incorrect Predictions
            </div>
        </div>""", unsafe_allow_html=True)

        # Key Factors
        tags_html = "".join([f'<span class="tag">{t}</span>' for t in top5_display])
        st.markdown(f"""
        <div class="tags-container">
            <h3>🔑 Key Factors</h3>
            {tags_html}
        </div>""", unsafe_allow_html=True)

        # About the Model
        st.markdown("""
        <div class="about-card" style="margin-top:16px;">
            <h3>🎯 About the Model</h3>
            <p>
                <b>Model:</b> XGBoost Classifier<br><br>
                This model predicts whether an individual earns more than $50K per year
                based on demographic and employment data.
            </p>
        </div>""", unsafe_allow_html=True)

        # Dataset Info
        st.markdown(f"""
        <div class="dataset-card">
            <h3>📁 Dataset Info</h3>
            <ul>
                <li><b>Source:</b> UCI Adult Dataset</li>
                <li><b>Records:</b> {raw_df.shape[0]:,}</li>
                <li><b>Features:</b> {raw_df.shape[1] - 1}</li>
                <li><b>Target:</b> Income &gt;$50K</li>
            </ul>
        </div>""", unsafe_allow_html=True)

        # Prediction result (shown after submit)
        if submitted:
            result = predict_income(
                age=age, edu_num=edu_num, hours=hours,
                cap_gain=cap_gain, marital_status=marital,
                model=model, scaler=scaler, feature_names=feature_names,
            )
            if result['prediction'] == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h2>💰 Income &gt; $50K</h2>
                    <p style="color:white;margin:8px 0 0 0;opacity:0.9">
                        Confidence: {result['prob_high']*100:.1f}%
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h2>💵 Income ≤ $50K</h2>
                    <p style="color:white;margin:8px 0 0 0;opacity:0.9">
                        Confidence: {result['prob_low']*100:.1f}%
                    </p>
                </div>""", unsafe_allow_html=True)

            st.progress(result['prob_high'], text=f">50K → {result['prob_high']*100:.1f}%")
            st.progress(result['prob_low'],  text=f"≤50K → {result['prob_low']*100:.1f}%")

    st.markdown("""
    <div class="footer">
        Built with ❤️ using Streamlit | Income Prediction App<br>
        Model trained on UCI Adult Census Income dataset
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("""
    <div class="hero-banner" style="padding:28px 32px;">
        <h1 style="font-size:1.8rem">📊 Exploratory Data Analysis</h1>
        <p>Explore the Adult Census dataset visually</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            edu_opts = sorted(raw_df['education'].dropna().unique().tolist())
            edu_filter = st.multiselect("Education", edu_opts, default=edu_opts)
        with fc2:
            age_min, age_max = int(raw_df['age'].min()), int(raw_df['age'].max())
            age_eda = st.slider("Age Range", age_min, age_max,
                                (age_min, age_max), key="eda_age")
        with fc3:
            occ_opts = ["All"] + sorted(raw_df['occupation'].dropna().unique().tolist())
            occ_filter = st.selectbox("Occupation", occ_opts)

    df_eda = raw_df[
        raw_df['education'].isin(edu_filter) &
        raw_df['age'].between(age_eda[0], age_eda[1])
    ]
    if occ_filter != "All":
        df_eda = df_eda[df_eda['occupation'] == occ_filter]

    st.caption(f"📌 Filtered: **{len(df_eda):,}** rows")

    t1, t2, t3, t4 = st.tabs(
        ["🎯 Target", "📈 Distributions", "📦 Boxplots", "🚨 Outliers"]
    )
    with t1:
        st.pyplot(plot_target_distribution(df_eda))
    with t2:
        st.pyplot(plot_numeric_distributions(df_eda))
    with t3:
        st.pyplot(plot_boxplots_by_income(df_eda))
    with t4:
        df_enc = handle_missing_and_duplicates(df_eda.copy())
        df_enc = encode_features(df_enc)
        report_out = detect_outliers(df_enc)
        fig_out = plot_outliers(report_out)
        if fig_out:
            st.pyplot(fig_out)
            st.dataframe(
                pd.DataFrame(list(report_out.items()), columns=['Column', 'Outlier Count'])
                .sort_values('Outlier Count', ascending=False),
                use_container_width=True,
            )
        else:
            st.success("✅ No outliers found in filtered data!")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="hero-banner" style="padding:28px 32px;">
        <h1 style="font-size:1.8rem">🤖 Model Performance</h1>
        <p>XGBoost Classifier — detailed evaluation metrics</p>
    </div>""", unsafe_allow_html=True)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, lbl, val in zip(
        [mc1, mc2, mc3, mc4, mc5],
        ["ACCURACY", "F1 SCORE", "AUC", "PRECISION", "RECALL"],
        [f"{acc:.1%}", f"{f1:.4f}", f"{metrics['roc_auc']:.4f}", f"{prec:.1%}", f"{recall:.1%}"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value" style="font-size:1.5rem">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    nb1, nb2, nb3 = st.columns(3)
    nb1.metric("🎲 Naive Accuracy",    f"{naive['accuracy']:.2%}")
    nb2.metric("🎲 Naive F1 Score",    f"{naive['f1']:.4f}")
    nb3.metric("🚀 XGBoost Improvement",
               f"+{(acc - naive['accuracy']):.2%}", delta_color="normal")

    st.divider()

    t1, t2, t3, t4 = st.tabs(["🧩 Confusion Matrix", "📈 ROC Curve",
                               "🔍 Feature Importance", "⭐ Top-5 Features"])
    with t1:
        st.pyplot(plot_confusion_matrix(y_test.values, y_pred))
    with t2:
        st.pyplot(plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc']))
    with t3:
        top_n = st.slider("Top N Features", 5, 30, 15)
        st.pyplot(plot_feature_importance(model, feature_names, top_n=top_n))
    with t4:
        st.subheader("⭐ Top 5 Most Important Features")
        for i, (feat, disp) in enumerate(zip(top5_features, top5_display), 1):
            st.markdown(f"**{i}.** {disp}  `({feat})`")
        st.info("💡 هذه الـ 5 features بتحمل أعلى importance score في الموديل.")

    st.divider()
    st.subheader("📋 Classification Report")
    report_df = pd.DataFrame(metrics['report']).T.drop(index=['accuracy'], errors='ignore')
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Overview
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner" style="padding:28px 32px;">
        <h1 style="font-size:1.8rem">🏠 Dataset Overview</h1>
        <p>UCI Adult Census Dataset — summary and interactive explorer</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Rows",  f"{raw_df.shape[0]:,}")
    c2.metric("🔢 Features",    f"{raw_df.shape[1] - 1}")
    c3.metric("⬆️ >50K",        f"{(raw_df['income'] == '>50K').sum():,}")
    c4.metric("⬇️ <=50K",       f"{(raw_df['income'] == '<=50K').sum():,}")

    st.divider()
    st.subheader("📄 Raw Data — Interactive Filter")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        income_filter = st.selectbox("Income", ["All", "<=50K", ">50K"])
    with col_f2:
        age_min2, age_max2 = int(raw_df['age'].min()), int(raw_df['age'].max())
        age_range = st.slider("Age Range", age_min2, age_max2, (age_min2, age_max2))
    with col_f3:
        wc_opts = ["All"] + sorted(raw_df['workclass'].dropna().unique().tolist())
        wc_filter = st.selectbox("Workclass", wc_opts)

    filtered = raw_df.copy()
    if income_filter != "All":
        filtered = filtered[filtered['income'] == income_filter]
    filtered = filtered[filtered['age'].between(age_range[0], age_range[1])]
    if wc_filter != "All":
        filtered = filtered[filtered['workclass'] == wc_filter]

    st.caption(f"Showing **{len(filtered):,}** rows")
    st.dataframe(filtered.head(300), use_container_width=True)
