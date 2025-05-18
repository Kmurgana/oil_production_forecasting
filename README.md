# oil\_production\_forecasting

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

Forecasting oil production using advanced time series models. 

---

## 📖 Overview

**oil\_production\_forecasting** is a data science project structured with [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) best practices. The main goal is to model, analyze, and forecast oil production in the Gulf Coast (PADD 3) region of the United States using robust statistical and machine learning techniques. 

---

## 📊 Dataset Description

This project uses historical monthly crude oil production data for **PADD 3 (Gulf Coast region)**, sourced from the **U.S. Energy Information Administration (EIA)**. The dataset spans from **1960 to present**, enabling detailed trend, seasonality, and anomaly analyses for field-level production. 

### **Key Features**

* **MBBL/D** (Thousand Barrels per Day)
* **MBBL** (Thousand Barrels per Month)
* Coverage: Texas, Louisiana, Mississippi, Alabama, New Mexico (offshore)
* Frequency: Monthly 

### **Data Columns**

| Column               | Description                            |             |
| -------------------- | -------------------------------------- | ----------- |
| `period`             | Reporting date (YYYY-MM-DD, monthly)   |             |
| `duoarea`            | DUO Area code (R30 for PADD 3)         |             |
| `area-name`          | Region name (PADD 3 - Gulf Coast)      |             |
| `product`            | Product code                           |             |
| `product-name`       | Product (Crude Oil)                    |             |
| `process`            | Process code                           |             |
| `process-name`       | Process description (Field Production) |             |
| `series`             | Series code                            |             |
| `series-description` | Data series details                    |             |
| `value`              | Production value (MBBL/D or MBBL)      |             |
| `units`              | Measurement units (`MBBL/D`, `MBBL`)   |  |

### **Data Source**

* **Agency:** U.S. Energy Information Administration (EIA)
* **Dataset:** Petroleum Supply Monthly – Crude Oil Production by PADD
* **Region:** PADD 3 (Gulf Coast: TX, LA, MS, AL, NM offshore)
* **Time Range:** 1960 – Present
* **Frequency:** Monthly 

### **Use Cases**

* Historical production trend analysis
* Time series forecasting (Prophet, ARIMA, LSTM)
* Outlier & anomaly detection
* Upstream oil & gas economic/operational decision support 

---

## 🗂️ Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   ├── cleaned        <- Cleaned dataset from the first phase.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── results                
│   ├── arima_predictions.csv        <- Arima results
│   ├── arima_results.json           <- Arima best parameters
│   ├── lstm_predictions.csv         <- LSTM results
│   ├── lstm_predictions.json        <- Arima best parameters
│   ├── prophet_predictions.csv      <- Prophet results
│   ├── prophet_predictions.json     <- Prophet best parameters
│   └─── model_metrics_summary.csv   <- Summary of all the best parameters.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         oil_production_forecasting and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── oil_production_forecasting  <- Source code for use in this project.
    ├── api
    │   ├── templates                     
    │   │   └── index.html      <- User interface made on html.
    │   └ app.py                <- Main aplication script.
    │
    ├─── modeling                
    │    ├── train.py           <- Script for training the LSTM model.
    │    └── predict.py         <- Script for creating the prediction.
    │
    ├── __init__.py             <- Makes oil_production_forecasting a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── pipeline.py             <- Pipeline that orchestates all the codes.
    │
    └─── modeling                
        ├── __init__.py 
        ├── predict.py          <- Code to run model inference with trained models          
        └── train.py            <- Code to train models

```
---

## 🚀 Getting Started

### **Prerequisites**

* Python 3.8+
* [poetry](https://python-poetry.org/) or `pip` for dependency management
* Recommended: virtualenv 

### **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/oil_production_forecasting.git
   cd oil_production_forecasting
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *or, if using poetry:*

   ```bash
   poetry install
   ```

---

## 🏃 Usage

### **Running the API**

Make sure you are in the main directory. For example:

```bash
cd C:\Users\kmurg\Desktop\Kevin\challenges\upstream\oil_production_forecasting
```

Run the app:

```bash
python -m oil_production_forecasting.api.app
```


### **Running Tests**

```bash
python -m pytest --cov=oil_production_forecasting tests/
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request. 

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information. 

---

If you want more details or advanced usage instructions, just ask! 
