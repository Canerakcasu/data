import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
try:
    from umap.umap_ import UMAP
    _USE_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    _USE_UMAP = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # Suppress causal inference warnings

# File list (assumed to be in the same folder)
# 2023 files have one structure, 2025 file has different structure
files_2023 = ['1_1_2023_general.csv', '1_3_2023_general.csv', '1_6_2023_general.csv']
file_2025 = 'upwork_jobs_6_2025.csv'

# Load and combine 2023 datasets
try:
    df_list_2023 = [pd.read_csv(f) for f in files_2023]
except Exception:
    # Try tab delimiter (fallback for earlier Colab errors)
    df_list_2023 = [pd.read_csv(f, delimiter='\t', encoding='utf-8', errors='ignore') for f in files_2023]

df_2023_raw = pd.concat(df_list_2023, ignore_index=True)

# Load 2025 dataset
df_2025_raw = pd.read_csv(file_2025)

# Process 2023 data
df_2023_raw.drop_duplicates(subset=['uid'], inplace=True) 
df_2023_raw['job_hourly_budget'] = pd.to_numeric(df_2023_raw['hourly_rate'], errors='coerce')
df_2023_raw.dropna(subset=['job_hourly_budget', 'client_payment_verification_status'], inplace=True)
df_2023_raw['job_publish_time'] = pd.to_datetime(df_2023_raw['published_on'], errors='coerce')
# Filter only 2023 data
df_2023 = df_2023_raw[df_2023_raw['job_publish_time'].dt.year == 2023].copy()

# Process 2025 data
df_2025_raw.drop_duplicates(subset=['id'], inplace=True)
df_2025_raw['job_hourly_budget'] = (pd.to_numeric(df_2025_raw['job_hourly_budget_min'], errors='coerce') + 
                                     pd.to_numeric(df_2025_raw['job_hourly_budget_max'], errors='coerce')) / 2
# Rename to match 2023 column name for consistency
df_2025_raw.rename(columns={'client_payment_verified': 'client_payment_verification_status'}, inplace=True)
df_2025_raw.dropna(subset=['job_hourly_budget', 'client_payment_verification_status'], inplace=True)
df_2025_raw['job_publish_time'] = pd.to_datetime(df_2025_raw['job_publish_time'], errors='coerce')
df_2025 = df_2025_raw.copy()

print(f"Loaded data: 2023 ({len(df_2023)} rows), 2025 ({len(df_2025)} rows)") 

def prepare_ate_data_psm(data, treatment_val=1.0, name='', year=2023):
    # Define covariates based on year (different columns in 2023 vs 2025)
    if year == 2023:
        X_cols = ['tier', 'client_location_country', 'type', 'engagement'] 
        country_col = 'client_location_country'
    else:  # 2025
        X_cols = ['job_contractor_tier', 'client_country', 'job_type', 'job_engagement_duration']
        country_col = 'client_country'
    
    # 1. Clean missing values
    data.dropna(subset=['job_hourly_budget', 'client_payment_verification_status'] + X_cols, inplace=True)
    
    # 2. Critical fix: Filter rare country categories (stabilizes matrix)
    top_n_countries = data[country_col].value_counts().nlargest(10).index
    data = data[data[country_col].isin(top_n_countries)].copy()

    # 3. Treatment and outcome
    data['T'] = (data['client_payment_verification_status'] == treatment_val).astype(int)
    data['Y'] = data['job_hourly_budget']
    
    # 4. OHE with drop_first=True to prevent multicollinearity
    X = pd.get_dummies(data[X_cols], drop_first=True)
    
    # Check for sufficient treatment/control groups to avoid errors
    if data['T'].sum() == 0 or (len(data['T']) - data['T'].sum()) == 0:
        ds_name = name if name else 'data'
        print(f"WARNING: {ds_name} dataset has insufficient Treated/Control groups for PSM.")
        return np.array([]), np.array([]), np.array([]), []
        
    return data['Y'].values, data['T'].values, X.values, list(X.columns)
def run_psm(data_year, Y, T, X, X_cols):
    if len(Y) < 100 or T.sum() == 0 or (len(T) - T.sum()) == 0:
        print(f"PSM WARNING ({data_year}): Insufficient observations/treatment groups. Results may not be reliable.")
        return {'ATE': np.nan, 'CI_low': np.nan, 'CI_high': np.nan}
    
    # Use OLS regression to estimate treatment effect (alternative to PSM)
    # Model: Y = beta_0 + beta_1*T + beta_2*X + error
    # ATE is captured by beta_1 (coefficient on treatment)
    try:
        # Combine treatment and covariates
        X_with_treatment = np.column_stack([T, X])
        X_reg = sm.add_constant(X_with_treatment)
        model = sm.OLS(Y, X_reg).fit()
        
        # The coefficient on T (index 1 after constant) is the ATE
        ate_result = model.params[1]
        ci_low = model.conf_int()[1, 0]
        ci_high = model.conf_int()[1, 1]
        
        return {'ATE': ate_result, 'CI_low': ci_low, 'CI_high': ci_high}
    except Exception as e:
        print(f"Error in treatment effect estimation for {data_year}:", repr(e))
        return {'ATE': np.nan, 'CI_low': np.nan, 'CI_high': np.nan}


# Causal Inference (ATE) - Compare 2023 vs 2025
Y_23, T_23, X_23, _ = prepare_ate_data_psm(df_2023, treatment_val=1.0, name='2023', year=2023)
Y_25, T_25, X_25, _ = prepare_ate_data_psm(df_2025, treatment_val=1.0, name='2025', year=2025)

results_23 = run_psm("2023", Y_23, T_23, X_23, [])
results_25 = run_psm("2025", Y_25, T_25, X_25, [])

print("--- Causal Inference (ATE) - 2023 vs 2025 Comparison ---")
print(f"2023 ATE: {results_23['ATE']:.4f} USD/hour (95% CI: [{results_23['CI_low']:.4f}, {results_23['CI_high']:.4f}])")
print(f"2025 ATE: {results_25['ATE']:.4f} USD/hour (95% CI: [{results_25['CI_low']:.4f}, {results_25['CI_high']:.4f}])")
print(f"Change from 2023 to 2025: {results_25['ATE'] - results_23['ATE']:.4f} USD/hour")


# Quantile Regression (2023 vs 2025 comparison)
print("\n--- Quantile Regression (75th Percentile) - 2023 vs 2025 ---")

for year_name, df_year in [('2023', df_2023), ('2025', df_2025)]:
    df_q = df_year.copy()
    
    # Check which columns exist in this year's data
    if 'tier' in df_q.columns:
        df_q.dropna(subset=['job_hourly_budget', 'tier', 'engagement', 'type'], inplace=True)
        df_q['is_expert'] = (df_q['tier'] == 'Expert').astype(int)
        df_q['long_term'] = (df_q['engagement'].str.contains('30+', na=False)).astype(int)
    else:
        df_q.dropna(subset=['job_hourly_budget', 'job_contractor_tier', 'job_engagement_duration', 'job_type'], inplace=True)
        df_q['is_expert'] = (df_q['job_contractor_tier'] == 'Expert').astype(int)
        df_q['long_term'] = (df_q['job_engagement_duration'].str.contains('More than 6 months|6 months', na=False)).astype(int)

    # Check data size: skip model if insufficient observations.
    MIN_QREG_OBS = 30
    if len(df_q) < MIN_QREG_OBS:
        print(f"{year_name}: Insufficient data ({len(df_q)} observations). Skipping.")
    else:
        X_q = sm.add_constant(df_q[['is_expert', 'long_term']])
        y_q = df_q['job_hourly_budget']
        try:
            model_q = QuantReg(y_q, X_q).fit(q=0.75, max_iter=5000)

            expert_coef = model_q.params['is_expert']
            longterm_coef = model_q.params['long_term']

            print(f"{year_name}: Expert Tier Coefficient: {expert_coef:.4f} USD/hour, Long-term Engagement: {longterm_coef:.4f} USD/hour")
        except Exception as e:
            print(f"{year_name}: Error in quantile regression - {repr(e)}")

# Text Analysis and UMAP - Both 2023 and 2025
print("\n--- Text Analysis & Embedding ---")

for year_name, df_year in [('2023', df_2023), ('2025', df_2025)]:
    # Check which columns exist
    if 'title' in df_year.columns and 'tier' in df_year.columns:
        required_text_cols = {'title', 'job_hourly_budget', 'tier'}
        tier_col = 'tier'
    elif 'title' in df_year.columns and 'job_contractor_tier' in df_year.columns:
        required_text_cols = {'title', 'job_hourly_budget', 'job_contractor_tier'}
        tier_col = 'job_contractor_tier'
    else:
        print(f"{year_name}: Missing required columns. Skipping text analysis.")
        continue
    
    missing_text_cols = required_text_cols - set(df_year.columns)

    if missing_text_cols:
        missing_list = ', '.join(sorted(missing_text_cols))
        print(f"{year_name}: Missing column(s): {missing_list}. Skipping.")
    else:
        df_umap = df_year.copy()
        df_umap.dropna(subset=list(required_text_cols), inplace=True)

        if df_umap.empty or len(df_umap) < 2:
            print(f"{year_name}: Insufficient observations for embedding. Skipping.")
        else:
            # Use title column
            text_data = df_umap['title'].fillna('')

            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform(text_data)

            if tfidf_matrix.shape[0] < 2:
                print(f"{year_name}: Too few observations. Skipping.")
            else:
                if _USE_UMAP:
                    umap_model = UMAP(n_components=2, random_state=42, metric='cosine')
                    embedding = umap_model.fit_transform(tfidf_matrix)
                else:
                    print(f"{year_name}: Using TSNE (umap-learn not installed).")
                    tfidf_dense = tfidf_matrix.toarray()
                    tsne = TSNE(n_components=2, random_state=42, metric='cosine')
                    embedding = tsne.fit_transform(tfidf_dense)

                df_umap['UMAP_X'] = embedding[:, 0]
                df_umap['UMAP_Y'] = embedding[:, 1]

                # Scatter Plot
                plt.figure(figsize=(12, 8))
                sns.scatterplot(
                    data=df_umap,
                    x='UMAP_X',
                    y='UMAP_Y',
                    hue=tier_col,
                    size='job_hourly_budget',
                    sizes=(20, 200),
                    alpha=0.6,
                    palette='viridis'
                )
                plt.title(f'Embedding of Job Titles - {year_name} (Sized by Budget, Colored by Tier)')
                plt.legend(title='Job Tier', loc='upper right')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.tight_layout()
                plt.savefig(f'embedding_plot_{year_name}.png', dpi=300)
                print(f"{year_name}: Text embedding plot saved as 'embedding_plot_{year_name}.png'")