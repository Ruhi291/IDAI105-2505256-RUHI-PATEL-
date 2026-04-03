# IDAI105-2505256-RUHI-PATEL-
SmartCharging Analytics is an interactive EV charging station dashboard built with Streamlit. It performs data cleaning, exploratory data analysis, K-Means clustering, association rule mining, and anomaly detection on EV charging data. The app uncovers behavior patterns, identifies unusual stations, and provides actionable business recommendations.
# ⚡ SmartCharging Analytics — Uncovering EV Behavior Patterns

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Status](https://img.shields.io/badge/Status-Deployed-green)
![Course](https://img.shields.io/badge/Course-Data%20Mining%20Year%201-orange)

STREAMLIT APP LINK: https://arfqb2b3tv69uvyijbczef.streamlit.app/


---

## 📋 Project Overview

**SmartCharging Analytics** is an interactive data mining dashboard built using Python and Streamlit. This project is developed as part of the **Data Mining Year 1 Summative Assessment — Scenario 2: SmartCharging Analytics Uncovering EV Behavior Patterns**.

The project simulates the role of a data analyst at the **SmartEnergy Data Lab**, working with EV (Electric Vehicle) charging infrastructure providers. The goal is to analyze EV charging station data to improve station utilization, customer experience, and infrastructure planning through advanced data mining techniques.

---

## 🎯 Project Objectives

| Objective | Description |
|-----------|-------------|
| 🔵 Cluster Charging Behaviors | Group stations based on usage, capacity, cost, and availability |
| 🚨 Detect Anomalies | Identify unusual consumption behaviors, faulty stations, or overuse |
| 🔗 Discover Associations | Explore relationships between charging features and demand |
| 🏗️ Infrastructure Planning | Support decisions on where and when to expand charging stations |
| 📊 Deploy Dashboard | Interactive Streamlit app for exploring patterns and anomalies |

---

## 📂 Repository Structure

```
SMART-EV-CHARGING/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)
```

---

## 📊 Dataset Description

The dataset contains **500 EV charging station records** with the following columns:

| Column | Description |
|--------|-------------|
| Station_ID | Unique identifier for each charging station |
| Latitude / Longitude | Geographic coordinates of the station |
| Address | Physical address of the station |
| Charger_Type | Type of charger — AC Level 1, AC Level 2, or DC Fast |
| Cost_USD_kWh | Cost of charging per kilowatt-hour in USD |
| Availability | Percentage uptime availability of the station |
| Distance_to_City_km | Distance from the station to the nearest city in km |
| Usage_Stats_avg_users_day | Average number of users per day |
| Station_Operator | Company operating the station (ChargePoint, Tesla, Blink, EVgo, Greenlots) |
| Charging_Capacity_kW | Maximum charging power in kilowatts |
| Connector_Types | Type of connector available at the station |
| Installation_Year | Year the station was installed |
| Renewable_Energy_Source | Whether the station uses renewable energy (Yes or No) |
| Reviews_Rating | Customer rating from 1 to 5 |
| Parking_Spots | Number of parking spots at the station |
| Maintenance_Frequency | How often the station is maintained |

> **Note:** The dataset is synthetically generated within the app using NumPy. It closely mirrors real-world EV charging station data structure and behavior patterns. No external dataset file is required.

---

## 🔧 Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | 1.32+ | Web dashboard and UI |
| Pandas | 2.0+ | Data manipulation and analysis |
| NumPy | 1.26+ | Numerical computing and data generation |
| Matplotlib | 3.8+ | Static charts and visualizations |
| Seaborn | 0.13+ | Statistical data visualization |
| Plotly | 5.18+ | Interactive charts and graphs |
| Scikit-learn | 1.4+ | K-Means clustering, PCA, preprocessing |
| SciPy | 1.12+ | Statistical methods for anomaly detection |

---

## 🚀 How to Run the App Locally

### Step 1 — Install Python
Download and install **Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/).
Make sure to check **"Add Python to PATH"** during installation.

### Step 2 — Clone or Download the Repository
```bash
git clone https://github.com/YourUsername/YourRepositoryName
cd YourRepositoryName
```

### Step 3 — Install Required Libraries
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy
```

### Step 4 — Run the App
```bash
streamlit run app.py
```

### Step 5 — Open in Browser
The app will automatically open at:
```
http://localhost:8501
```

---

## 📱 App Features — All 7 Stages

---

### 🏠 Stage 1 — Home & Project Scope

- Displays key metrics: total stations, average daily users, renewable stations, anomaly count
- Lists all project objectives clearly
- Shows the full dataset column reference guide
- Displays a sample of the raw dataset

---

### 🧹 Stage 2 — Data Cleaning & Preprocessing

The following preprocessing steps are applied to ensure data quality:

- **Duplicate Removal** — Removed duplicate records based on Station_ID
- **Missing Value Handling:**
  - Numeric columns (Reviews_Rating) → filled with median value
  - Categorical columns (Renewable_Energy_Source) → filled with default value
  - Categorical columns (Connector_Types) → filled with mode (most common value)
- **Label Encoding** — Converted categorical columns to numeric values for machine learning:
  - Charger_Type, Station_Operator, Renewable_Energy_Source, Connector_Types, Maintenance_Frequency
- **StandardScaler Normalization** — Normalized continuous variables to the same scale:
  - Cost_USD_kWh, Usage_Stats_avg_users_day, Charging_Capacity_kW, Distance_to_City_km, Reviews_Rating, Availability, Parking_Spots

**Visualizations:**
- Missing values handling summary with color-coded cards
- Distribution histograms for all continuous variables
- Correlation heatmap showing relationships between key features
- Full cleaned dataset preview table

---

### 📊 Stage 3 — Exploratory Data Analysis (EDA)

Five comprehensive tabs covering all EDA requirements:

**Usage Distribution:**
- Histogram showing distribution of average daily users
- Boxplot comparing usage across charger types

**Charger Type Analysis:**
- Bar chart showing station count by charger type
- Bar chart showing average charging capacity by type
- Demand heatmap showing usage across Charger Type vs Availability bands

**Cost by Operator:**
- Box plots showing cost distribution per kWh for each operator

**Ratings & Usage:**
- Scatter plot of Reviews Rating vs Average Daily Usage
- Renewable energy impact analysis table

**Growth Over Time:**
- Dual-axis chart showing stations installed per year alongside average daily users

---

### 🔵 Stage 4 — Clustering Analysis (K-Means)

**Features used for clustering:**
- Usage_Stats_avg_users_day
- Charging_Capacity_kW
- Cost_USD_kWh
- Availability

**Three tabs:**

**Elbow Method:**
- Inertia plot to find the optimal number of clusters
- Silhouette score plot for validation
- Best K is highlighted automatically

**Cluster Scatter Plot:**
- PCA (Principal Component Analysis) reduces features to 2D
- Interactive scatter plot colored by cluster label
- Hover details show station info

**Cluster Profiles:**
- Table showing average feature values per cluster
- Bar chart showing station count per cluster by charger type

**Cluster Labels:**
| Label | Description |
|-------|-------------|
| 🟢 Occasional Users | Low usage, low capacity, low cost stations |
| 🟡 Daily Commuters | Moderate usage, medium capacity stations |
| 🔴 Heavy Users | High usage, high capacity, DC Fast dominated |

---

### 🔗 Stage 5 — Association Rule Mining

**Method:** Manual Frequent Itemset Mining — equivalent to the Apriori algorithm. No external library required.

**Features discretized for mining:**
- Charger_Type
- Renewable_Energy_Source
- Usage_Level (Low / Medium / High)
- Cost_Level (Low / Medium / High)
- Capacity_Level (Low / Mid / High)
- Maintenance_Frequency

**Metrics calculated:**
| Metric | Description |
|--------|-------------|
| Support | How frequently the itemset appears in the dataset |
| Confidence | How often the rule is correct |
| Lift | How much more likely the consequent is given the antecedent |

**Three tabs:**
- Rules Table — top 25 rules sorted by Lift
- Lift Bar Chart — visual top 15 rules
- Support vs Confidence scatter plot with Lift as bubble size

**Key Finding:** DC Fast Charger combined with High Usage Level is a consistently strong association. Low-cost stations near city centers reliably show high demand.

---

### 🚨 Stage 6 — Anomaly Detection

Two statistical methods used to detect anomalies:

**IQR Method (Interquartile Range):**
- Flags stations where Usage > Q3 + 1.5 × IQR
- Interactive boxplot showing normal vs anomalous stations in red

**Z-Score Method:**
- Flags stations with |Z-score| > 3
- Scatter plot showing all station Z-scores with threshold line

**High Cost + Low Rating Detection:**
- Separately flags stations that charge high prices but receive poor reviews
- These are candidates for service quality audit

**Three tabs:**
- IQR Boxplot
- Z-Score Analysis with High Cost Low Rating scatter
- Full anomaly records table with all flagged stations

---

### 💡 Stage 7 — Insights & Reporting

**Summary Findings:**
- Clustering results and what each cluster represents
- Most popular charger type and its average usage
- Renewable energy impact on daily users
- Number and percentage of anomalies detected
- Key association rules discovered
- Urban vs rural usage comparison

**Operator Comparison:**
- Summary table with average rating, usage, cost, and station count per operator
- Scatter plot comparing operators on rating vs usage

**Business Recommendations:**

| # | Recommendation |
|---|---------------|
| 1 | Expand DC Fast Charger network in Heavy User cluster zones |
| 2 | Audit high-cost low-rating stations for service quality issues |
| 3 | Increase renewable energy share across the network |
| 4 | Investigate anomalous stations for faults or capacity needs |
| 5 | Focus expansion strategy on underserved rural areas |
| 6 | Align maintenance schedules with peak usage periods |

---

## 📈 Key Conclusions

After completing all stages of data mining analysis on the EV charging station dataset, the following key conclusions were drawn:

1. **DC Fast chargers are the most in-demand** — They generate the highest average daily usage across all station types and dominate the Heavy Users cluster identified by K-Means clustering.

2. **Renewable energy stations attract more users** — Stations powered by renewable energy sources show a measurable increase in average daily usage compared to non-renewable stations, suggesting that sustainability is valued by EV users.

3. **Urban stations are significantly busier than rural ones** — Stations located within 10 km of a city center average considerably more daily users than rural stations, confirming that geographic proximity to urban areas is a major driver of demand.

4. **Anomalies represent real business opportunities** — The anomalous stations detected through IQR and Z-score methods are not just data errors. Many represent high-demand locations that are underserved and need capacity expansion urgently.

5. **Association rules reveal reliable demand patterns** — Strong associations between DC Fast chargers, high capacity, and high usage levels can guide infrastructure investment decisions. Low-cost stations near cities consistently attract high footfall.

6. **Operator quality varies significantly** — The operator comparison reveals clear differences in pricing and customer satisfaction. Some operators charge premium rates but receive poor reviews, indicating a disconnect between pricing and service quality.

7. **Maintenance frequency impacts availability** — Stations with monthly maintenance schedules show higher availability percentages, directly linking maintenance investment to station uptime and user experience.

---




---

## 👤 Author

- **Student Name:** Ruhi Patel 
- **Student ID:** 2505256
- **Course:** Data Mining — Year 1
- **CRS:** Artificial Intelligence
- **Assessment:** Summative Individual Project
- **Scenario:** Scenario 2 — SmartCharging Analytics

---

## 📚 References

- Streamlit Documentation — https://docs.streamlit.io
- Scikit-learn Clustering — https://scikit-learn.org
- Plotly Python Charts — https://plotly.com/python
- K-Means Clustering Guide — https://neptune.ai/blog/k-means-clustering
- Anomaly Detection Techniques — https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html
- EV Charging Behavior Research — https://arxiv.org/pdf/1802.04193
- Association Rule Mining — https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/

---

*Built with ❤️ using Python and Streamlit — Data Mining Year 1 Summative Assessment*
