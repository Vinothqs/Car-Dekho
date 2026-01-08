# 🚗 CarDekho – Stable Production Model Monitoring

## Overview

This repository demonstrates a **stable production-style model monitoring implementation** for the **CarDekho Used Car Price Prediction** machine learning project.

The goal of this monitoring system is to ensure that the deployed ML model **continues to behave reliably over time**, even as real-world data, market trends, and user behavior change.

This implementation is intentionally designed to be:
- Beginner-friendly
- Easy to understand
- Aligned with real-world production practices
- Extendable to enterprise ML systems

---

## What is Model Monitoring?

Model monitoring is the **post-deployment process** of continuously checking whether a machine learning model is:

- Receiving valid and complete input data  
- Producing reasonable and stable predictions  
- Operating within expected statistical boundaries  

Monitoring does **not** retrain or modify the model.  
Instead, it **observes, measures, and reports** model behavior so that informed decisions can be made.

---

## Why Model Monitoring is Important for CarDekho

The CarDekho model predicts **used car prices** based on features such as:
- Car age  
- Kilometers driven  
- Fuel type  
- Transmission  
- Seller type  

In real-world scenarios:
- Market prices change  
- New car variants appear  
- Customer behavior evolves  
- Data pipelines can fail or degrade  

Without monitoring, a model can continue running while silently producing **unrealistic or risky predictions**.

Model monitoring acts as an **early warning system**.

---

## Monitoring Scope (Version 1)

This project implements **Phase 1 – Foundational Production Monitoring**.

### Included in this version
- Data volume monitoring  
- Data quality monitoring  
- Prediction behavior monitoring  

### Not included (intentionally)
- Automatic model retraining  
- Advanced drift metrics (PSI, KS)  
- Performance monitoring using real labels  

**Principle followed:**  
Start simple → stabilize → then expand

---

## High-Level Monitoring Architecture

```
Deployed ML Model
        ↓
 New Prediction Data
        ↓
 Python Monitoring Script
        ↓
 Monitoring Metrics (CSV / Table)
        ↓
 Review & Decision Making
```

Monitoring runs independently of the prediction pipeline.

---

## Project Structure

```
car_dekho_monitoring/
│
├── monitoring/
│   ├── monitor_price_model.py
│   ├── thresholds.py
│   └── utils.py
│
├── data/
│   ├── predictions_latest.csv
│   └── predictions_history.csv
│
├── outputs/
│   └── monitoring_results.csv
│
├── README.md
```

---

## Monitoring Implementation – Step by Step

### Step 1: Define Monitoring Inputs

The monitoring script consumes:
- Latest prediction data (current run)
- Historical prediction data (baseline reference)

Monitoring **only reads data** and never modifies model outputs.

---

### Step 2: Define Monitoring Metrics

**Data Volume Metrics**
- Total number of records
- Comparison with previous runs

**Data Quality Metrics**
- Missing value percentage
- Invalid value detection

**Prediction Behavior Metrics**
- Min / Max predicted price
- Average predicted price
- High-price percentage

---

### Step 3: Define Threshold Rules

Example rules:
- Row count drop > 30% → ALERT  
- Missing values > 10% → WARNING  
- Large change in average predicted price → ALERT  

Thresholds are configurable and tunable.

---

### Step 4: Monitoring Script Logic

The monitoring job follows a fixed process:
1. Load latest predictions
2. Load historical baseline
3. Calculate metrics
4. Evaluate thresholds
5. Assign status (OK / WARNING / ALERT)
6. Save results

---

### Simplified Pseudo Code

```python
load_latest_predictions()
load_historical_predictions()

metrics = calculate_metrics()
results = evaluate_against_thresholds(metrics)

save_monitoring_results(results)
```

---

### Step 5: Monitoring Output

Example output format:

| run_date | metric_name | metric_value | status |
|--------|------------|-------------|--------|
| 2026-01-10 | row_count | 1020 | OK |
| 2026-01-10 | avg_pred_price | 5.4L | OK |
| 2026-01-10 | high_price_pct | 45% | ALERT |

---

## How Monitoring is Used in Practice

1. Monitoring job runs on schedule  
2. Results are reviewed  
3. Alerts trigger investigation  
4. Decision is made (retrain, fix data, or no action)  

Monitoring **supports decisions**, it does not automate them.

---

## Model Training vs Model Monitoring

| Aspect | Model Training | Model Monitoring |
|------|---------------|----------------|
| When | Before deployment | After deployment |
| Purpose | Learn patterns | Detect issues |
| Frequency | Occasional | Continuous |
| Output | Trained model | Monitoring report |

---

## Future Enhancements

- Feature drift detection  
- Performance monitoring  
- Dashboards and alerts  

The core monitoring structure remains unchanged.

---

## Key Takeaways

- Monitoring is continuous  
- Python scripts are the foundation  
- Start simple and scale gradually  
- Outputs must be auditable  

---

## Final Note

This project demonstrates how **production-style model monitoring** can be implemented using the same principles followed in enterprise ML systems.
