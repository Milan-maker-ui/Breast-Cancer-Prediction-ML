# 🩺 Breast Cancer Prediction Using Machine Learning

## 📌 Project Overview

Breast Cancer Prediction is a Machine Learning project that classifies breast tumor samples into two categories:

- **Benign:** Non-cancerous tumor
- **Malignant:** Cancerous tumor

This project uses the Breast Cancer Wisconsin Diagnostic Dataset and a Decision Tree Classifier to predict tumor categories.

The main objective is to understand machine learning classification, data analysis, visualization, model training, and evaluation using Python.

> ⚠️ **Disclaimer:** This project is developed for educational purposes only. It is not a clinically validated medical diagnostic system and must not be used for medical decisions.

## 🎯 Project Objectives

- Load and explore the breast cancer dataset.
- Perform exploratory data analysis (EDA).
- Analyze feature distributions and correlations.
- Train a Decision Tree classification model.
- Predict benign and malignant tumor categories.
- Evaluate model performance using different metrics.
- Visualize model results and feature importance.

## 📊 Dataset Information

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:** Scikit-learn built-in dataset.

| Dataset Details | Information |
|---|---|
| Total Samples | 569 |
| Total Features | 30 |
| Target Classes | 2 |
| Problem Type | Binary Classification |
| Missing Values | 0 in the supplied dataset |

### Target Classes

| Target Value | Diagnosis |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The dataset contains numerical features computed from digitized images of breast mass samples.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 🧠 Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks.

**Model Configuration:**

| Parameter | Value |
|---|---|
| Algorithm | Decision Tree |
| Criterion | Gini |
| Maximum Depth | 4 |
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Data Split | Stratified |

## 📁 Project Structure

```text
Breast-Cancer-Prediction-ML/
│
├── notebooks/
│   └── breast_cancer_prediction.ipynb
│
├── src/
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

The `outputs/` folder containing evaluation plots is generated automatically when you run `train.py`.

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Milan-maker-ui/Breast-Cancer-Prediction-ML.git
```

### Step 2: Navigate to the project folder

```bash
cd Breast-Cancer-Prediction-ML
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ How to Run the Project

### Option 1: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

`notebooks/breast_cancer_prediction.ipynb`

Run all cells sequentially.

### Option 2: Run Python Script

```bash
python src/train.py
```

The script trains the model, evaluates predictions, and saves visualization images.

## 📈 Model Evaluation

The model is evaluated using the following performance metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures the proportion of predicted positives that are correct |
| Recall | Measures the proportion of actual positives correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect classifications |

Actual performance scores are calculated when the project is executed.

## 📊 Data Visualization

The notebook includes:

1. Diagnosis distribution visualization.
2. Feature distribution histograms.
3. Feature correlation heatmap.
4. Confusion matrix.
5. Top 10 feature importance visualization.

The Python training script saves:

- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`

## 🚀 Future Improvements

- Compare Decision Tree with Random Forest.
- Implement Logistic Regression and Support Vector Machine.
- Perform hyperparameter tuning.
- Apply cross-validation.
- Build an interactive web application using Flask or Streamlit.
- Implement explainable AI techniques.

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Python programming.
- Data preprocessing and analysis.
- Exploratory Data Analysis (EDA).
- Machine Learning classification.
- Decision Tree algorithms.
- Model evaluation and visualization.
- Scikit-learn implementation.

## 👨‍💻 Author

**Milan Arote**

- GitHub: [Milan-maker-ui](https://github.com/Milan-maker-ui)
- LinkedIn: [Milan Arote](https://www.linkedin.com/in/milan-arote-0652872a5/)

## ⭐ Acknowledgements

- [Scikit-learn](https://scikit-learn.org/)
- [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- Python open-source community

---

**Project Category:** Artificial Intelligence & Machine Learning

**Purpose:** Educational and portfolio project.
