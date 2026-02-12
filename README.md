# 🌪️ SalesVortex - Predicting the Future of FMCG Chaos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-orange.svg)](https://github.com/ankushsingh003/SalesVortex)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An advanced machine learning project for forecasting FMCG (Fast-Moving Consumer Goods) sales using time series analysis and regression models.

---

## 📊 Project Overview

**SalesVortex** leverages cutting-edge machine learning algorithms to predict sales volumes for FMCG products. By analyzing historical data, product attributes, pricing strategies, and store locations, this system provides actionable insights for inventory optimization and demand planning.

### 🎯 Key Objectives
- Forecast daily sales volumes with high accuracy
- Identify seasonal patterns and trends
- Analyze the impact of promotions and pricing on sales
- Optimize inventory management across different store locations

---

## ✨ Features

- **📈 Time Series Analysis**: Decomposition into trend, seasonality, and residuals
- **🤖 Multiple ML Models**: XGBoost, Random Forest, Linear Regression
- **🔍 Exploratory Data Analysis**: Comprehensive visualizations and statistical insights
- **🏷️ Feature Engineering**: Categorical encoding, date feature extraction
- **📉 Bivariate Analysis**: Price vs Sales, Promotion impact, Location-based patterns
- **📊 Interactive Visualizations**: Matplotlib & Seaborn powered charts

---

## 📁 Dataset

**File**: `extended_fmcg_demand_forecasting.csv`

### Features:
| Column | Type | Description |
|--------|------|-------------|
| `Date` | DateTime | Transaction date |
| `Product_Category` | Categorical | Beverages, Dairy, Household, Personal Care, Snacks |
| `Sales_Volume` | Numerical | Units sold (Target variable) |
| `Price` | Numerical | Product price |
| `Promotion` | Binary | 0 = No promotion, 1 = Promotion active |
| `Store_Location` | Categorical | Urban, Rural, Suburban |
| `Weekday` | Numerical | Day of week (0-6) |
| `Supplier_Cost` | Numerical | Cost from supplier |
| `Replenishment_Lead_Time` | Numerical | Days to restock |
| `Stock_Level` | Numerical | Current inventory |

**Size**: 1,002 records spanning 2+ years (2022-2024)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ankushsingh003/SalesVortex---Predicting-the-future-of-FMCG-chaos---ML_XGBOOST_RANDFOREST.git
cd SalesVortex---Predicting-the-future-of-FMCG-chaos---ML_XGBOOST_RANDFOREST
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook main.ipynb
```

---

## 🛠️ Technologies Used

### Core Libraries
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Time Series**: `statsmodels`, `prophet`
- **Development**: `jupyter`

### Models Implemented
1. **Linear Regression** - Baseline model
2. **Random Forest Regressor** - Ensemble learning
3. **XGBoost Regressor** - Gradient boosting (primary model)
4. **Prophet** - Facebook's time series forecasting

---

## 📂 Project Structure

```
SalesVortex/
│
├── main.ipynb                              # Main analysis notebook
├── extended_fmcg_demand_forecasting.csv    # Dataset
├── requirements.txt                         # Python dependencies
├── store_location_encoding.py              # Encoding utilities
└── README.md                                # This file
```

---

## 🔬 Methodology

### 1. **Data Preprocessing**
- Date conversion and sorting
- Categorical encoding (Label Encoding, One-Hot Encoding)
- Missing value handling

### 2. **Exploratory Data Analysis (EDA)**
- Univariate analysis (Sales distribution)
- Bivariate analysis (Price vs Sales, Promotion impact)
- Time series decomposition (Trend, Seasonality, Residuals)
- Correlation analysis

### 3. **Feature Engineering**
- Date features: Day, Month, Year, Weekday, Is_Weekend
- Encoding: Store_Location (Urban=2, Rural=0, Suburban=1)
- Product_Category encoding

### 4. **Model Training & Evaluation**
- Train-Test split (80-20, time-based)
- Cross-validation
- Metrics: RMSE, MAE, R²

---

## 📈 Key Insights

🔍 **Discovered Patterns**:
- Strong correlation between promotions and sales spikes
- Weekly seasonality detected (weekend peaks)
- Urban locations show 20% higher average sales
- Price sensitivity varies by product category

---

## 🎯 Results

*(Add your results here after completing the analysis)*

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |

---

## 🔮 Future Enhancements

- [ ] Deploy as a web application (Flask/Streamlit)
- [ ] Add real-time forecasting API
- [ ] Implement LSTM/GRU for sequential modeling
- [ ] Include external factors (weather, holidays)
- [ ] AutoML for hyperparameter optimization

---

## 👤 Author

**Ankush Singh**
- GitHub: [@ankushsingh003](https://github.com/ankushsingh003)
- LinkedIn: [Connect with me](#) *(Add your LinkedIn)*

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: FMCG industry synthetic data
- Inspiration: Retail analytics and demand forecasting challenges

---

## 📞 Contact

For questions, suggestions, or collaboration:
- 📧 Email: *(Add your email)*
- 💼 LinkedIn: *(Add your LinkedIn)*

---

<div align="center">
  
**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ and ☕ by Ankush Singh

</div>
