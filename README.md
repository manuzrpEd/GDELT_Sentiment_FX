# GDELT 2.0 FX Sentiment Alpha  
**Zero look-ahead, event-timed news sentiment engine for systematic macro FX trading**

**Author**: Manuel Antonio Sánchez García  
PhD Economics
manuel.sanchez.garcia@protonmail.com · +34 644 402 872
LinkedIn: [linkedin.com/in/manuel-sanchez-garcia](https://www.linkedin.com/in/manuel-antonio-sanchez-garcia-6b3a7a136/)

---

### Project Overview

Independent research project developing a **production-grade, zero-look-ahead news sentiment alpha** for 25+ currencies (EM + G10) using the full GDELT 2.0 Event Database (2018–present).

This is a **full-stack systematic macro research pipeline** built to institutional standards.

**Currently in active development** — final results to be published upon completion.

---

### Key Features

- **Zero look-ahead bias**: Only same-day events (`SQLDATE == publication date`)
- **Event-timed processing**: News impact aligned with exact publication timestamp
- **Full GDELT 2.0 coverage**: 15+ emerging + G10 currencies (TRY, ZAR, BRL, MXN, INR, IDR, PLN, HUF, CLP, EGP, NGN, EUR, GBP, JPY, etc.)
- **Accurate country mapping**: ISO-3166-1 alpha-3 + special handling for EUR
- **Universal XGBoost model** trained in long format → learns cross-sectional sentiment regimes across all currencies
- **Daily cross-sectional ranking** → long top-N / short bottom-N predicted-return basket
- **Rigorous walk-forward framework**:
  - Training: 2018–2021
  - Validation: 2022–2023 (hyperparameter & basket size selection)
  - OOS: 2024–present (strictly untouched during development)

---

### Tech Stack

- **Python** (pandas, numpy, XGBoost, scikit-learn)
- **Yahoo Finance API** (daily FX prices)
- **VectorBT Pro** (institutional backtesting)
- **GDELT 2.0 HTTP API** (full historical event stream via direct CSV downloads)
- **Parquet** for efficient storage
- **Git** + modular structure (src/, data/, notebooks/)

---

### Why This Matters

This project demonstrates — **entirely independently** — the use of **alternative data + machine learning alpha** used for tier-1 hedge funds and investment banks.

It demonstrates:
- Deep understanding of **news sentiment as a macro predictor**
- Production-grade **data engineering & feature engineering** with alternative datasets
- Institutional **research discipline** (walk-forward, no overfitting, clean OOS)
- Ability to deliver **live, client-facing tools** (Jupyter notebooks)

---

### Status: Work in Progress

Currently finalizing:
- Walk-forward hyperparameter optimization
- Robustness checks
- Transaction cost modeling and position sizing refinements

**Out-of-sample performance (2024–present) under strict no-peeking protocol** — results will be published upon completion.

---