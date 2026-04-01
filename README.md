# 🏥 Insurance Premium Prediction – ML Internship Project

**Student:** Sanjay Thakor | **Roll No:** 220390107031
**Course:** Machine Learning Internship | **Guide:** Prof. Akshay Kansara

---

## 📁 Project Structure

```
├── 1_EDA_Analysis.ipynb          # EDA on 1,338 records
├── 2_Model_Training.ipynb        # Model training (small dataset)
├── streamlit_app.py              # Streamlit dashboard (small dataset)
├── insurance (1).csv             # Dataset (1,338 records)
├── best_model.pkl                # Trained Gradient Boosting model
├── encoders.pkl                  # Label encoders
├── model_info.pkl                # Model metadata
├── requirements.txt              # Dependencies
│
└── large_dataset_project/
    ├── 3_Large_EDA_Analysis.ipynb       # EDA on 50,000 records
    ├── 4_Large_Model_Training.ipynb     # Model training (large dataset)
    ├── streamlit_app_large.py           # Streamlit dashboard (large dataset)
    ├── large_insurance_50000.csv        # Dataset (50,000 records)
    ├── best_model_large.pkl             # Trained model (large)
    ├── encoders_large.pkl               # Encoders (large)
    └── model_info_large.pkl             # Model metadata (large)
```

## 🚀 Run the App

```bash
# Small dataset
streamlit run streamlit_app.py

# Large dataset
cd large_dataset_project
streamlit run streamlit_app_large.py
```

## 📊 Results

| Dataset | Best Model | Accuracy | MAE |
|---------|-----------|----------|-----|
| Small (1,338) | Gradient Boosting | 87.67% | $2,425 |
| Large (50,000) | Gradient Boosting | 93.11% | $1,974 |

## 🛠️ Tech Stack
Python · Pandas · Scikit-learn · Streamlit · Plotly
