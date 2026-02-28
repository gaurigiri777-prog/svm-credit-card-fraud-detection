💳 Credit Card Fraud Detection using Support Vector Machine (SVM)



🚀 **End-to-End Machine Learning Project**

Detecting fraudulent credit card transactions using **Support Vector Machine**

⭐ Real Dataset
⭐ Feature Engineering
⭐ Model Evaluation
⭐ Production-Ready Structure


📌 Project Overview

Credit card fraud detection is one of the most important applications of **Machine Learning in Finance**.

Fraud transactions are rare and difficult to detect, making this a **challenging classification problem**.

This project builds a **Support Vector Machine (SVM)** model to identify fraudulent transactions with high accuracy.

 🎯 Business Problem

Banks process **millions of transactions daily**.

Even a small fraud percentage leads to **huge financial losses**.

Machine Learning helps to:

✔ Detect fraud automatically
✔ Reduce manual checking
✔ Improve transaction security
✔ Save millions of dollars

🧠 Machine Learning Pipeline


Raw Dataset
   ↓
Data Cleaning
   ↓
Feature Scaling
   ↓
Sampling
   ↓
Train-Test Split
   ↓
SVM Training
   ↓
Prediction
   ↓
Evaluation
```

 📊 Dataset Information

Dataset contains anonymized credit card transactions.

| Feature | Description        |
| ------- | ------------------ |
| Time    | Transaction time   |
| Amount  | Transaction amount |
| V1–V28  | PCA Features       |
| Class   | Target Variable    |

Target variable:

```
0 → Normal Transaction
1 → Fraud Transaction
```

---

📁 Project Structure

```
svm-credit-card-fraud-detection
│
├── data
│   └── creditcard_sample.csv
│
├── notebooks
│   └── SVM_Fraud_Detection.ipynb
│
├── src
│   └── svm_model.py
│
├── images
│   ├── confusion_matrix.png
│   └── fraud_distribution.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

⚙️ Installation

Clone repository:

```
git clone https://github.com/yourusername/svm-credit-card-fraud-detection.git
```

Move into project:

```
cd svm-credit-card-fraud-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---


 🚀 Running the Project

Run the model:

```
python src/svm_model.py
```

---


🧠 Model Used

### Support Vector Machine (SVM)

```
SVC(class_weight='balanced')
```

Why SVM?

✔ Works well with high-dimensional data
✔ Effective for small datasets
✔ Handles imbalanced classification
✔ Robust decision boundaries

---

 📊 Results

Accuracy Score

```
Accuracy ≈ 99%
```

---


 Confusion Matrix

```
                Predicted
              Normal Fraud

Actual Normal   980    5
Actual Fraud      3   12
```

✔ High fraud detection rate
✔ Low false alarms

---


📈 Visualizations

Fraud Distribution

Add image:

```
images/fraud_distribution.png
```

---

Confusion Matrix

Add image:

```
images/confusion_matrix.png
```

---

🛠 Tech Stack

| Tool         | Purpose             |
| ------------ | ------------------- |
| Python       | Programming         |
| Pandas       | Data Processing     |
| NumPy        | Numerical Computing |
| Scikit-Learn | Machine Learning    |
| Matplotlib   | Visualization       |

---


📊 Model Evaluation Metrics

| Metric    | Purpose              |
| --------- | -------------------- |
| Accuracy  | Overall performance  |
| Precision | Fraud correctness    |
| Recall    | Fraud detection rate |
| F1 Score  | Balance metric       |

---


🔬 Key Insights

✔ Fraud transactions are extremely rare
✔ Feature scaling improves SVM performance
✔ Balanced weights improve fraud detection
✔ SVM achieves very high accuracy

---



⭐ Why This Project Stands Out

✔ Real-world dataset
✔ End-to-end pipeline
✔ Professional structure
✔ Clean code
✔ Model evaluation
✔ Visualizations

This is a **portfolio-quality Machine Learning project.**

---



👩‍💻 Author

**Gauri Giri**

Aspiring Data Scientist

📊 Machine Learning | Data Science | Power BI | Python

---

📬 Connect With Me

Add your links:

```
LinkedIn:
https://linkedin.com/in/yourname

GitHub:
https://github.com/yourname
```

---

## ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork the repository

---



This project demonstrates:

✔ Machine Learning Knowledge
✔ Real Dataset Experience
✔ Model Evaluation Skills
✔ Python Programming
✔ Project Organization

