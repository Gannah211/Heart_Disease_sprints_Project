# Heart Disease Prediction Project

## 📌 Overview
This project implements a **full end-to-end machine learning pipeline** for predicting the risk of heart disease using the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).  
It includes:
- Data preprocessing & cleaning
- Dimensionality reduction (PCA)
- Feature selection (Random Forest, RFE, Chi-Square)
- Supervised models (Logistic Regression, Decision Tree, Random Forest, SVM)
- Unsupervised models (K-Means, Hierarchical Clustering)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Exporting the final model as `.pkl`
- A **Streamlit** web app for real-time prediction

---

## 📂 Project Structure

Heart_Disease_Project/ <br>
│── data/<br>
│ ├── heart_disease.csv<br>
│<br>
│── notebooks/<br>
│ ├── 01_data_preprocessing.ipynb<br>
│ ├── 02_pca_analysis.ipynb<br>
│ ├── 03_feature_selection.ipynb<br>
│ ├── 04_supervised_learning.ipynb<br>
│ ├── 05_unsupervised_learning.ipynb<br>
│ ├── 06_hyperparameter_tuning.ipynb<br>
│ ├── 07_model_export.ipynb<br>
│ ├──model_meta.json<br>
│ ├──pca_heart_disease.csv<br>
│<br>
│── models/<br>
│ ├── final_model.pkl<br>
│ <br>
│<br>
│── ui/<br>
│ ├── streamlit_app.py<br>
│<br>
│── results/<br>
│ ├── evaluation_metrics.txt<br>
│<br>
│── requirements.txt<br>
│── README.md<br>
│── .gitignore<br>

---

---

## 📊 Dataset
The dataset contains medical attributes of patients, including:
- Age, sex, chest pain type, resting blood pressure, cholesterol levels
- Fasting blood sugar, resting ECG results, maximum heart rate achieved
- Exercise-induced angina, ST depression, slope of the peak exercise ST segment
- Number of major vessels, thalassemia
- Target: `0` → No heart disease, `1` → Heart disease present

---

## ⚙️ Installation & Setup

```bash
# 1️⃣ Clone the repository 
git clone https://github.com/YOUR_USERNAME/Heart_Disease_Project.git


cd Heart_Disease_Project


python -m venv venv


# on Windows:
venv\Scripts\activate
on Mac/Linux:
source venv/bin/activate

# 5️⃣Install dependencies
pip install -r requirements.txt

```
🚀 Running the Project
1️⃣ Run the Jupyter notebooks
Follow the order in the notebooks/ folder:

01_data_preprocessing.ipynb → Data cleaning and scaling

02_pca_analysis.ipynb → Apply PCA and save results

03_feature_selection.ipynb → Select most important features

04_supervised_learning.ipynb → Train classification models & save metrics

05_unsupervised_learning.ipynb → Perform clustering

06_hyperparameter_tuning.ipynb → Optimize model performance

07_model_export.ipynb → Export final model as .pkl and save metadata

---

2️⃣ Run the Streamlit App

```bash
cd ui
streamlit run streamlit_app.py

```
The app allows manual entry of patient data and returns:

Probability of heart disease

Prediction: Disease / No Disease


🛠 Tools & Libraries
-Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

-Streamlit (Interactive UI)

-Joblib (Model saving)

-Jupyter Notebook (Analysis & development)

