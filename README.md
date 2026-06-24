# CharityML-Donor-Prediction-App 💰

### NTI Graduation Project

A Streamlit web app that predicts whether an individual is likely to be a donor based on their demographics, employment, and financial information. This project demonstrates machine learning deployment and interactive dashboard design using Python.

---

## ✨ Features
* Predict donor potential based on features such as **age, education, occupation, marital status, and income**.
* Interactive **Streamlit interface** with tabs for Demographics, Employment, and Financial information.
* Professional and compact model performance dashboard.
* Confidence score displayed for each prediction.
* Robust handling of **categorical variables** for XGBoost models.
* Compatible with older **XGBoost models** using a patch wrapper.

---

## 🛠️ Technologies Used
* **Python** (Core programming language)
* **Streamlit** (For building the interactive web application UI)
* **XGBoost** (Advanced Gradient Boosting model used for classification)
* **Scikit-Learn** (For data preprocessing and scaling pipelines)
* **Pandas & NumPy** (For data manipulation and analysis)
* **Jupyter Notebook** (For Exploratory Data Analysis & training experiments)

---

## 💻 Installation & Usage

1. **Clone the repository:**
```bash
   git clone [https://github.com/MohamedEslam111/Project_NTI.git](https://github.com/MohamedEslam111/Project_NTI.git)
   cd Project_NTI
1. Run the App:
streamlit run app.py

📂 Project Structure
project_nti/
│
└── project nti 1/
    ├── app.py                      # Main Streamlit application
    ├── main.py                     # Central execution script
    ├── model.py                    # Model definition and architecture
    ├── predict.py                  # Inference and prediction logic
    ├── preprocessing.py            # Data cleaning and feature engineering
    ├── config.py                   # Configuration and hyperparameter settings
    ├── Income_Prediction_FinalNTI.ipynb  # Jupyter Notebook for EDA & Training
    ├── adult                       # The dataset file
    ├── xgb_model.pkl               # Trained XGBoost model weights
    ├── scaler.pkl                  # Saved StandardScaler instance
    └── feature_columns.pkl         # Saved list of expected feature names

✉️ Contact
Mohamed Eslam Metwally Awaad
