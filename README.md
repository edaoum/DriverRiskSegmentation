# 🚗 Driver Risk Segmentation — Usage-Based Insurance

> **Unsupervised machine learning applied to telematics data for behavior-based insurance pricing**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 📌 Business Context

Traditional car insurance pricing relies on **static demographic variables** (age, gender, vehicle type) that are poor proxies for actual driving risk. Telematics technology now enables insurers to collect real-time behavioral data directly from vehicles.

This project applies **K-Means clustering** to driver telematics data to automatically identify distinct risk profiles. The resulting segments enable insurers to implement **Usage-Based Insurance (UBI)** — a data-driven pricing model where premiums reflect actual driving behavior rather than statistical demographics.

### Business Objective

> Segment drivers into homogeneous risk groups so that safe drivers are rewarded, high-risk drivers are correctly priced, and the insurer optimises its claims exposure.

---

## 🧩 Methodology — CRISP-DM

This project follows the industry-standard **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

```
1. Business Understanding  →  Define the pricing problem and segmentation goals
2. Data Understanding      →  EDA, distributions, correlations, outlier detection
3. Data Preparation        →  Missing value imputation, feature standardisation
4. Modeling                →  K-Means clustering with Elbow + Silhouette validation
5. Evaluation              →  Cluster profiling, silhouette analysis, radar charts
6. Deployment              →  Actionable recommendations per risk segment
```

---

## 📁 Repository Structure

```
DataMiningProject/
│
├── notebooks/
│   └── Driver_Risk_Segmentation.ipynb   # Main analysis notebook
│
├── data/
│   ├── driver_data.csv                  # Raw telematics dataset
│   └── driver_data_segmented.csv        # Output: dataset enriched with cluster labels
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🔬 What the Notebook Covers

| Section | Content |
|---------|---------|
| **1. Setup & Data Loading** | Imports, configuration, data ingestion |
| **2. Exploratory Data Analysis** | Distributions, correlation matrix, outlier detection |
| **3. Data Preprocessing** | Median imputation, StandardScaler normalisation |
| **4. Optimal k Selection** | Elbow method + silhouette analysis side by side |
| **5. K-Means Clustering** | Model fitting, PCA 2D projection, silhouette per sample |
| **6. Cluster Profiling** | Centroid heatmap, violin plots, radar chart fingerprints |
| **7. Business Recommendations** | Risk label per cluster, fleet composition, export |

---

## 📊 Example Output — Driver Risk Segments

| Segment | Profile | Recommended Action |
|---------|---------|-------------------|
| 🟢 Safe & Experienced | Low speed variance, consistent mileage, few incidents | Loyalty discount (−10% to −20%) |
| 🟡 Moderate Risk — Urban | Short trips, frequent braking, city driving patterns | Standard premium + telematics coaching |
| 🔴 High-Risk | Speed events, harsh acceleration, elevated incident rate | Risk surcharge (+15% to +30%) |
| 🔵 Occasional & Prudent | Low mileage, calm style, long gaps between trips | Pay-per-mile (PPM) product |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation and numerical computation |
| `scikit-learn` | K-Means clustering, StandardScaler, PCA, silhouette metrics |
| `matplotlib` / `seaborn` | Visualisations (distributions, heatmaps, radar, violin plots) |
| Jupyter Notebook | Interactive, narrative-driven analysis |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/edaoum/DataMiningProject.git
cd DataMiningProject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook notebooks/Driver_Risk_Segmentation.ipynb
```

---

## 📈 Key Results

- **Clustering method**: K-Means with automatic `k` selection via silhouette score
- **Dimensionality reduction**: PCA for cluster visualisation (2D projection)
- **Validation**: Silhouette analysis both at aggregate and individual sample level
- **Output**: Segmented dataset ready for CRM integration, actuarial modelling, or BI dashboards

---

## 🔭 Limitations & Future Work

- **Cluster stability**: validate across multiple random seeds using ensemble clustering
- **Temporal drift**: driver behaviour evolves — consider periodic re-segmentation
- **Supervised validation**: if claims history is available, correlate clusters with actual loss ratios to confirm risk ordering
- **Feature enrichment**: add time-of-day, road type, and weather conditions for finer-grained profiling

---

---

*Academic project — Data Mining module. Analysis is fully reproducible from the provided dataset and `requirements.txt`.*
