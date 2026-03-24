# 🛍️ Mall Customer Segmentation Analysis

> A Machine Learning project that segments mall customers into behavioral groups using K-Means clustering — helping businesses craft targeted marketing strategies.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project analyzes a mall customer dataset (90,000 records) and groups customers into distinct segments based on:
- **Annual Income**
- **Spending Score**
- **Age & Gender**
- **Visit Frequency & Annual Spend**

Clusters are identified using **K-Means** and predictions are powered by a **Random Forest Classifier** — exposed via a simple **Flask web app**.

---

## 🗂️ Project Structure

```
mall-customer/
│
├── data/
│   └── Mall_customers.csv         # Dataset (90,000 records, 7 features)
│
├── notebooks/
│   ├── mall_customer_analysis.ipynb  # Full EDA + Clustering notebook
│   └── app.ipynb                     # App prototype notebook
│
├── src/
│   ├── analysis.py                # Core ML logic (clustering, models)
│   └── predict.py                 # Prediction helper functions
│
├── templates/
│   ├── home.html                  # Input page
│   └── result.html                # Output / results page
│
├── app.py                         # Flask application entry point
├── requirements.txt               # Python dependencies
├── Procfile                       # Deployment config (Heroku/Render)
└── README.md
```

---

## 📊 Dataset

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Age` | Customer age |
| `Gender` | Male / Female |
| `Annual Income (k$)` | Annual income in thousands |
| `Spending Score (1-100)` | Mall-assigned spending score |
| `Annual_Spend` | Total annual spend amount |
| `Visit_Frequency` | Number of visits |

- **Rows:** 90,000
- **Missing values:** None ✅

---

## 🧠 Machine Learning Pipeline

```
Raw Data → EDA → Feature Selection → Scaling → K-Means Clustering
                                                     ↓
                              Random Forest Classifier (for prediction)
                                                     ↓
                                          Flask Web App (live prediction)
```

### Models Used
| Model | Purpose |
|---|---|
| K-Means (k=5) | Customer segmentation |
| Random Forest Classifier | Predict cluster for new customers |
| Linear Regression | Spending score prediction |
| Decision Tree Regressor | Baseline comparison model |

### Customer Segments (5 Clusters)
| Cluster | Profile |
|---|---|
| 0 | Low income, Low spending |
| 1 | High income, Low spending |
| 2 | Medium income, Medium spending |
| 3 | Low income, High spending |
| 4 | High income, High spending ⭐ (Target) |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/mall-customer.git
cd mall-customer
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

### 5. Open in Browser
```
http://127.0.0.1:5000/
```

---

## 🧪 Run the Analysis Notebook

```bash
jupyter notebook notebooks/mall_customer_analysis.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Flask |
| ML / Data | scikit-learn, pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |
| Deployment | Gunicorn + Heroku/Render |

---

## 📈 Key Insights

- Customers with **high income + high spending score** are the most valuable segment
- **Young customers (18–30)** tend to have higher spending scores regardless of income
- **Visit frequency** shows moderate correlation with annual spend
- K-Means with **k=5** gives the most interpretable segments (confirmed by Elbow Method)

---

## ☁️ Deployment

This project is configured for deployment on **Heroku** or **Render** using the included `Procfile`:

```
web: gunicorn app:app
```

---

## 🔮 Future Improvements

- [ ] Add Elbow Method plot to auto-select optimal `k`
- [ ] Deploy on cloud (Render / Railway)
- [ ] Build interactive dashboard (Plotly / Streamlit)
- [ ] Add DBSCAN / Hierarchical clustering comparison
- [ ] REST API endpoint for batch predictions

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Durgam Vani**  
🔗 [github.com/Durgamvani-184](https://github.com/Durgamvani-184)
