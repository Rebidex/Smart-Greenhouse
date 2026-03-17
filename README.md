# 🌿 Smart Greenhouse Management and Environmental Monitoring Using IoT

> **Course project** — Knowledge-Based Systems  
> **Author:** Radu-Andrei Ugran

---

## Overview

This project proposes an intelligent, data-driven system for the autonomous management of a smart greenhouse. It integrates IoT sensor data with Machine Learning algorithms to optimize resource consumption (water, nutrients) while maintaining safe climate conditions for plant growth.

The system is built on a **hybrid architecture** that combines:
- A **reactive control loop** — real-time rule-based actuator control (IF-THEN logic)
- A **predictive optimization loop** — ML regression models that anticipate environmental needs

---

## Table of Contents

- [Motivation](#motivation)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Resource Optimization Logic](#resource-optimization-logic)
- [Results](#results)
- [Conclusions & Future Work](#conclusions--future-work)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Motivation

Modern agriculture faces a multidimensional crisis at the intersection of demographic pressure and environmental degradation. According to the FAO, global food production must increase by ~70% by 2050 to feed a projected population of 9.7 billion, yet arable land is shrinking and freshwater is becoming increasingly scarce.

Most commercial greenhouse solutions rely on **open-loop control systems** with rigid timers (e.g., *"irrigate for 10 minutes at 8:00 AM"*). These systems ignore the actual state of the plant and soil, leading to:

- **Over-irrigation** → water waste, nutrient leaching, fungal diseases
- **Under-irrigation** → water stress, stunted growth, crop loss

This project implements a shift toward **Agriculture 4.0** — Cyber-Physical Systems where physical processes are monitored and governed by integrated computational algorithms. The greenhouse becomes an autonomous agent that can *sense*, *reason*, and *act*.

---

## Objectives

1. **Reactive Monitoring & Control** — Develop a logic module that processes continuous sensor streams and issues instant ON/OFF commands to actuators (fans, pumps), maintaining climate parameters within safe thresholds without requiring human supervision.

2. **Resource Optimization** — Implement a decision algorithm that dynamically adjusts water and nutrient delivery based on synergistic factors: current temperature, atmospheric humidity, and existing soil nutrient levels.

3. **Predictive Environmental Modeling** — Train and compare three ML regression models (Linear, Polynomial, Random Forest) to predict critical parameter evolution and demonstrate that learning-based systems outperform static rule-based systems.

---

## Dataset

**Source:** *IoT Smart Greenhouse Dataset* — collected at Tikrit University, Iraq (available on Kaggle)

| Property | Details |
|---|---|
| Total records | 37,923 |
| Type | Time-series (continuous) |
| Features | 9 columns |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `Temperature` | Continuous | Indoor greenhouse temperature (°C), range ~10–40°C |
| `Humidity` | Continuous | Relative air humidity (%) |
| `Water_Level` | Continuous | Available water in the irrigation system (0–100%) |
| `N`, `P`, `K` | Continuous | Soil macronutrient concentrations (digital scale 0–255) |
| `Fan_actuator_ON` | Binary | Fan state at time of measurement |
| `Watering_plant_pump_ON` | Binary | Plant watering pump state |
| `Water_pump_actuator_ON` | Binary | Water supply pump state |

### Exploratory Data Analysis (EDA)

Key observations from the time-series analysis:

- **Diurnal periodicity** — Clear day/night oscillation cycles in temperature and humidity, confirming data validity and the need for temporal features in the model.
- **Inverse temperature–humidity correlation** — At temperature peaks, humidity drops significantly (physically consistent with increased vapor-holding capacity of warm air and ventilation).
- **Nutrient dynamics** — Periods of stability interrupted by abrupt changes, corresponding to fertilization or intensive irrigation events.

### Feature Correlation

The Pearson correlation matrix revealed:
- **Strong positive correlation between N, P, K** — nutrients vary together (likely applied as compound NPK fertilizer), enabling dimensionality reduction.
- **Significant negative temperature–humidity correlation** — exploited directly by the predictive models.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HYBRID CONTROL SYSTEM                   │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │   REACTIVE LOOP     │    │    PREDICTIVE LOOP        │   │
│  │  (Feedback Control) │    │  (Feedforward / ML)       │   │
│  │                     │    │                           │   │
│  │  IF temp > 36°C     │    │  Random Forest → Water   │   │
│  │  THEN Fan = ON      │    │  Polynomial Reg → Hum.   │   │
│  │                     │    │  Linear Reg → Temp.      │   │
│  │  IF water < 70%     │    │                           │   │
│  │  THEN Pump = ON     │    │  W_optim = W_target ×    │   │
│  │                     │    │  F_temp × F_hum × F_nutr │   │
│  └─────────────────────┘    └──────────────────────────┘   │
│                ↑                          ↑                  │
│         IoT Sensors              Historical Data             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing Pipeline

Raw IoT sensor data is often noisy, incomplete, or inconsistently scaled. A five-stage preprocessing pipeline was implemented:

### 1. Signal Denoising (Moving Average)
Transient sensor fluctuations are smoothed using a moving average filter with window size N=5:

$$y[t] = \frac{1}{N} \sum_{i=0}^{N-1} x[t-i]$$

Applied to all continuous variables: `temperature`, `humidity`, `water_level`, `N`, `P`, `K`.

### 2. Missing Values Imputation
IoT network failures can cause data gaps (`NaN`). **Mean Imputation** is used to fill missing values with the column mean, preserving time-series continuity.

### 3. Feature Engineering
The raw timestamp column is transformed into numeric temporal features:
- `Hour` — captures the diurnal cycle (day/night), correlated with temperature and photosynthesis
- `Day of Year` — captures seasonal variations

### 4. Chronological Train-Test Split
Because data is sequential, **random shuffling is avoided** (it would cause data leakage — the model would "see" the future). A strict chronological split is used:
- **Training set: 80%** (earliest records)
- **Test set: 20%** (latest records, simulating prediction of unknown future)

### 5. Standardization (Z-score)
Features have very different magnitudes (temperature ~30, nutrients ~200). Z-score normalization is applied via `StandardScaler`:

$$z = \frac{x - \mu}{\sigma}$$

All variables are brought to mean=0, variance=1.

### 6. Dimensionality Reduction (PCA)
Due to strong multicollinearity between N, P, and K, **Principal Component Analysis** is applied to retain **95% of original data variance** while eliminating redundancy and reducing overfitting risk.

---

## Machine Learning Models

Three regression models were trained using **Scikit-learn**, each matched to the dynamics of a specific target variable.

### Linear Regression — Temperature Prediction
Used as a baseline model. Assumes a linear relationship between input features (hour, previous humidity) and future temperature.

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \epsilon$$

### Polynomial Regression — Humidity Prediction
Humidity has a non-linear (parabolic) relationship with temperature. Input features are expanded using `PolynomialFeatures(degree=2)`:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon$$

### Random Forest Regressor — Water Level Prediction
The primary and most robust model. An ensemble of 100 decision trees, each making an independent prediction; the final output is the average.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |

**Validation:** 5-Fold Cross-Validation was used to ensure model performance is not dependent on a specific data split.

---

## Resource Optimization Logic

The optimization module computes a dynamic **Optimal Water Level** (W_optim), which adapts in real-time instead of using a fixed target:

$$W_{optim} = W_{target} \times F_{temp} \times F_{hum} \times F_{nutr}$$

| Factor | Description |
|---|---|
| `W_target` | Baseline ideal water level (85%) |
| `F_temp` | Temperature factor — increases above 1.0 when temp > 32°C to compensate for evapotranspiration |
| `F_hum` | Humidity factor — increases when humidity < 60% (dry air accelerates water loss) |
| `F_nutr` | Nutrient factor — decreases below 1.0 when soil is already nutrient-saturated, preventing over-dilution |

An **Efficiency Score** is also computed to quantify how close the current water level is to the calculated optimum.

### Actuator Control Rules

```python
# Fan control
if temperature > 36 or humidity > 75:
    fan = ON   # Cooling / dehumidification
else:
    fan = OFF

# Water supply pump
if water_level < 70:
    pump = ON  # Refill reservoir
else:
    pump = OFF
```

---

## Results

### Model Performance (Test Set — 20% of data)

| Model | Target Variable | RMSE | R² |
|---|---|---|---|
| Linear Regression | Temperature | — | — |
| Polynomial Regression | Humidity | — | — |
| **Random Forest** | **Water Level** | **lowest** | **~1.0** |

The **Random Forest Regressor** achieved the lowest RMSE and highest R² score (~1), confirming that Ensemble Learning methods are better suited for capturing the non-linear dynamics of biological systems compared to simple linear models.

### Resource Optimization Analysis

Simulation results demonstrated:
- The system correctly **increased water supply** during simulated "heatwave days" (temperature factor spikes visible in output graphs)
- The system correctly **reduced irrigation** during high-humidity periods, avoiding over-watering
- The hybrid architecture maintained system stability across all simulated scenarios

---

## Conclusions & Future Work

### Achieved Outcomes

1. **High Predictive Accuracy** — Random Forest validated with R² ≈ 1 on the test set, confirming the effectiveness of Ensemble Learning for complex biological system modeling.
2. **Dynamic Resource Optimization** — The system demonstrated measurable water savings by adapting to real-time conditions rather than fixed schedules.
3. **Operational Robustness** — The hybrid reactive+predictive architecture ensured both immediate safety (reactive loop) and long-term efficiency (predictive loop).

### Future Directions

1. **Deep Learning Integration** — Implement Feedforward Neural Networks (FNN) or Convolutional Neural Networks (CNN) for visual disease/pest detection from greenhouse cameras.
2. **Predictive Maintenance** — Collect additional actuator data (vibrations, current draw) to train failure-prediction models and reduce equipment downtime.
3. **Hardware Deployment (Edge Computing)** — Port the notebook code to a Raspberry Pi or NVIDIA Jetson to run ML inference directly in the greenhouse without internet dependency.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core implementation language |
| Jupyter Notebook | Development environment & documentation |
| Pandas | Data manipulation (37K+ records) |
| NumPy | Vectorized numerical operations |
| Scikit-learn | ML pipeline: preprocessing, training, cross-validation |
| Matplotlib / Seaborn | Data visualization & EDA |

---

## References

1. Food and Agriculture Organization of the United Nations (FAO), *"The future of food and agriculture — Trends and challenges,"* Rome, 2017. [Online]. Available: http://www.fao.org/3/a-i6583e.pdf

2. H. A. Ahmed, *"Smart Greenhouse IoT Dataset,"* Tikrit University, College of Computer Science and Mathematics, Iraq, 2023. [Online]. Available: https://www.kaggle.com/datasets

3. Scikit-learn Developers, *"Scikit-learn: Machine Learning in Python."* [Online]. Available: https://scikit-learn.org/stable/user_guide.html

4. Course and laboratory materials, *"Knowledge-Based Systems"* discipline, Technical University, current academic year.

5. The Pandas Development Team, *"pandas documentation."* [Online]. Available: https://pandas.pydata.org/docs/
