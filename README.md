#  BizLens Analytics: Enterprise Customer Retention System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.31-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![Status](https://img.shields.io/badge/Status-Development-green.svg)

> **Live Demo:** [Will be updated soon]

##  Executive Summary
**BizLens** is a vertical SaaS application designed to operationalize machine learning for customer retention. Unlike standard dashboards that simply visualize history, BizLens uses **XGBoost** to predict future churn risk and prescribes automated retention strategies.

It features a **strict-schema ETL layer** to handle real-world data variability and a **decision-support interface** tailored for non-technical stakeholders.

##  Key Features

### 1.  Auto-Schema ETL Wizard
Real-world client data is messy. BizLens includes a dedicated **ETL Adapter** (`src/etl_adapter.py`) that:
- Detects schema mismatches in uploaded CSVs.
- Provides a GUI wizard to map client columns to the system's internal logic.
- Standardizes data types for robust inference.

### 2.  Predictive Risk Engine
- **Algorithm:** XGBoost Classifier (Gradient Boosted Trees).
- **Pipeline:** Imbalanced-Learn pipeline with SMOTE (Synthetic Minority Over-sampling) to handle class imbalance.
- **Output:** Real-time probability scoring (0-100%) mapped to 4 Risk Tiers (Low, Medium, High, Critical).

### 3.  Automated Strategy Recommendations
The system doesn't just predict risk; it suggests action.
- **Critical Risk (>75%):** Recommends aggressive retention offers (e.g., 15% discount).
- **High Risk (50-75%):** Recommends service audits and bundle up-selling.

##  Technical Architecture

```mermaid
graph LR
    A[Client CSV] -->|Upload| B(ETL Adapter)
    B -->|Schema Validation| C{Valid?}
    C -- No --> D[Mapping Wizard UI]
    D --> B
    C -- Yes --> E[Preprocessing Pipeline]
    E -->|SMOTE + Scaling| F[XGBoost Inference]
    F --> G[Streamlit Dashboard]
