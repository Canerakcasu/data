# Upwork Job Market Analysis (2023 vs 2025)

A comprehensive statistical analysis comparing Upwork freelance job market data from 2023 and 2025, focusing on payment verification effects, tier-based pricing, and job clustering patterns.

## 📊 Overview

This project analyzes trends in the Upwork freelance marketplace by comparing job postings from 2023 and 2025. It employs causal inference, quantile regression, and text embedding techniques to understand how payment verification, contractor tiers, and engagement types affect hourly rates.

## 🎯 Key Analyses

### 1. **Causal Inference (Treatment Effect Analysis)**
- Estimates the Average Treatment Effect (ATE) of client payment verification on hourly budgets
- Uses OLS regression with covariate adjustment
- Compares effects across 2023 and 2025

### 2. **Quantile Regression (75th Percentile)**
- Analyzes the impact of Expert tier status and long-term engagements on high-budget jobs
- Shows how premium factors affect upper-quartile pricing
- Year-over-year comparison of coefficient changes

### 3. **Text Embedding & Visualization**
- TF-IDF vectorization of job titles
- UMAP/t-SNE dimensionality reduction for clustering
- Scatter plots showing job clusters by tier and budget

## 📁 Project Structure

```
data/
├── solution.py                    # Main analysis script
├── 1_1_2023_general.csv          # 2023 job data (January)
├── 1_3_2023_general.csv          # 2023 job data (March)
├── 1_6_2023_general.csv          # 2023 job data (June)
├── upwork_jobs_6_2025.csv        # 2025 job data (June)
├── embedding_plot_2023.png       # Generated visualization for 2023
├── embedding_plot_2025.png       # Generated visualization for 2025
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
```

**Optional (for UMAP - recommended):**
```bash
pip install umap-learn
```

> **Note**: If `umap-learn` is not installed, the script automatically falls back to t-SNE from scikit-learn.

## 💻 Usage

### Basic Execution

```bash
python solution.py
```

### Expected Output

```
Loaded data: 2023 (799 rows), 2025 (1881 rows)
--- Causal Inference (ATE) - 2023 vs 2025 Comparison ---
2023 ATE: -1.6710 USD/hour (95% CI: [-8.6903, 5.3483])
2025 ATE: -11.6587 USD/hour (95% CI: [-17.9068, -5.4106])
Change from 2023 to 2025: -9.9877 USD/hour

--- Quantile Regression (75th Percentile) - 2023 vs 2025 ---
2023: Expert Tier Coefficient: 15.0000 USD/hour, Long-term Engagement: 2.5000 USD/hour
2025: Expert Tier Coefficient: 0.0000 USD/hour, Long-term Engagement: -3.5000 USD/hour

--- Text Analysis & Embedding ---
2023: Text embedding plot saved as 'embedding_plot_2023.png'
2025: Text embedding plot saved as 'embedding_plot_2025.png'
```

### Output Files

The script generates:
- `embedding_plot_2023.png` - Scatter plot of job clusters for 2023
- `embedding_plot_2025.png` - Scatter plot of job clusters for 2025

## 📈 Key Findings

### Payment Verification Impact
- **2023**: Payment verification had minimal negative effect (-$1.67/hour)
- **2025**: Effect became significantly more negative (-$11.66/hour)
- **Change**: -$9.99/hour decrease, suggesting verified clients may be budget-conscious

### Expert Tier Premium
- **2023**: Expert contractors commanded +$15/hour at 75th percentile
- **2025**: Expert premium disappeared ($0/hour)
- **Insight**: Market commoditization - expertise matters less for high-budget jobs

### Long-term Engagement
- **2023**: Long-term projects paid +$2.50/hour more
- **2025**: Long-term projects paid -$3.50/hour less
- **Shift**: Preference moved toward short-term, flexible engagements

## 🔧 Technical Details

### Data Processing

**2023 Data:**
- Columns: `uid`, `hourly_rate`, `tier`, `client_location_country`, `type`, `engagement`, `published_on`, etc.
- Source: Combined from three CSV files (Jan, Mar, Jun 2023)

**2025 Data:**
- Columns: `id`, `job_hourly_budget_min`, `job_hourly_budget_max`, `job_contractor_tier`, `client_country`, `job_type`, `job_engagement_duration`, etc.
- Source: Single CSV file (Jun 2025)

### Methodology

1. **Data Cleaning**
   - Remove duplicates
   - Handle missing values
   - Standardize column names across years
   - Filter to top 10 countries to stabilize covariate matrix

2. **Causal Inference**
   - Treatment: Payment verification status (binary)
   - Outcome: Hourly budget (USD)
   - Method: OLS with covariate adjustment (tier, country, type, engagement)
   - Reports ATE with 95% confidence intervals

3. **Quantile Regression**
   - Target quantile: 75th percentile
   - Predictors: Expert tier (binary), Long-term engagement (binary)
   - Iterative solver (max 5000 iterations)

4. **Text Embedding**
   - Vectorization: TF-IDF (max 1000 features, English stop words)
   - Dimensionality reduction: UMAP (preferred) or t-SNE fallback
   - Visualization: Scatter plot colored by tier, sized by budget

## ⚠️ Limitations

- **Sample bias**: Only includes jobs with complete hourly budget and payment verification data
- **Confounding**: Treatment effect estimates assume no unmeasured confounders
- **Temporal**: 2023 data spans Jan-Jun; 2025 is June only
- **Generalization**: Results specific to Upwork platform

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📝 License

This project is for educational and research purposes.

## 👤 Author

**Caner Akcasu**
- GitHub: [@Canerakcasu](https://github.com/Canerakcasu)

## 📚 References

- Statsmodels Documentation: https://www.statsmodels.org/
- UMAP: Uniform Manifold Approximation and Projection: https://umap-learn.readthedocs.io/
- scikit-learn: https://scikit-learn.org/

---

*Last updated: October 31, 2025*
