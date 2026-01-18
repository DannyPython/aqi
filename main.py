import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import lognorm, norm, kstest, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT YOUR MASTER LOADER
try:
    import data
except ImportError:
    raise ImportError("❌ Could not import 'data.py'. Ensure both files are in the same folder.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FILTER_HAZE_SEASON = False 
N_SIMULATIONS = 10000

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("--- Loading Data ---")
try:
    df_clean = data.get_data()
except Exception as e:
    print(f"❌ Error loading data from data.py: {e}")
    exit()

# Optional: Seasonal Filtering
if FILTER_HAZE_SEASON:
    print("⚠️ FILTER ACTIVE: Using only Haze Season (June-Sept) data.")
    if 'date' in df_clean.columns:
        df_clean['month'] = df_clean['date'].dt.month
        df_clean = df_clean[df_clean['month'].isin([6, 7, 8, 9])]
        print(f"Data filtered to {len(df_clean)} rows.")
    else:
        print("⚠️ Warning: 'date' column missing. Skipping seasonal filter.")

# Define Model Variables
# Standardize names to match data.py output (lowercase)
target_var = 'pm25'
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 

# Safety check: Only keep columns that actually exist in the CSV
found_vars = [col for col in input_vars if col in df_clean.columns]
missing_vars = [col for col in input_vars if col not in df_clean.columns]

if missing_vars:
    print(f"⚠️ Warning: The following variables were not found in data: {missing_vars}")

if not found_vars:
    print("❌ Critical Error: No input variables found in dataset. Check column names in CSV.")
    exit()

if target_var not in df_clean.columns:
    print(f"❌ Critical Error: Target variable '{target_var}' not found.")
    exit()

print(f"Using inputs: {found_vars}")

# ==============================================================================
# 2. DETERMINISTIC MODEL (OLS Baseline)
# ==============================================================================
print("\n--- [Phase 1] Training Deterministic Model ---")
X = df_clean[found_vars]
Y = df_clean[target_var]
X_const = sm.add_constant(X)

# Fit OLS
try:
    model = sm.OLS(Y, X_const).fit()
    betas = model.params
    resid_std = model.resid.std()
    print(model.summary())
    print(f"\nModel Residual Std Dev (Noise): {resid_std:.2f}")
except Exception as e:
    print(f"❌ Error training OLS model: {e}")
    exit()

# ==============================================================================
# 3. DISTRIBUTION FITTING (Visual Validation)
# ==============================================================================
print("\n--- [Phase 2] Fitting Distributions (Visual Check) ---")
sns.set_theme(style="whitegrid", palette="muted")

num_vars = len(found_vars)
cols = 3
rows = (num_vars + cols - 1) // cols 

fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(15, 5 * rows), num="Figure 1: Distribution Fitting")
axes_dist = axes_dist.flatten()

fit_metrics = []

for i, col in enumerate(found_vars):
    # Handle zeros for Log-Normal (cannot take log of 0)
    data_col = df_clean[col].replace(0, 0.001)
    
    # Fit Log-Normal & Normal
    shape, loc, scale = lognorm.fit(data_col, floc=0)
    mu, std = norm.fit(data_col)
    
    # Calculate KS Error Statistics
    d_stat_log, _ = kstest(data_col, 'lognorm', args=(shape, loc, scale))
    d_stat_norm, _ = kstest(data_col, 'norm', args=(mu, std))
    
    fit_metrics.append({
        'Variable': col, 'LogNorm D-Stat': d_stat_log, 'Improvement %': (d_stat_norm - d_stat_log)/d_stat_norm * 100
    })
    
    # Plotting
    if i < len(axes_dist):
        ax = axes_dist[i]
        sns.histplot(data_col, stat="density", color="skyblue", alpha=0.5, label="Actual", ax=ax)
        x_range = np.linspace(data_col.min(), data_col.max(), 100)
        ax.plot(x_range, lognorm.pdf(x_range, shape, loc, scale), 'r-', lw=2, label=f"LogNorm")
        ax.plot(x_range, norm.pdf(x_range, mu, std), 'g--', lw=2, label=f"Normal")
        ax.set_title(f"{col.upper()} Fit")
        ax.legend()

for j in range(i + 1, len(axes_dist)):
    axes_dist[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)

print("\n[Distribution Metrics]")
print(pd.DataFrame(fit_metrics)[['Variable', 'LogNorm D-Stat', 'Improvement %']])

# ==============================================================================
# 4. MULTIVARIATE STOCHASTIC SIMULATION (Pro Upgrade)
# ==============================================================================
print("\n--- [Phase 3] Correlated Monte Carlo Simulation ---")

# A. Prepare Covariance (Log-Space)
# 'replace(0, 0.001)' prevents log(0) errors
log_data = np.log(df_clean[found_vars].replace(0, 0.001)) 
mu_log = log_data.mean()
cov_log = log_data.cov()

# B. Generate Correlated Random Variables (Robust Cholesky)
print(f"Generating {N_SIMULATIONS} scenarios...")
try:
    # 'allow_singular=True' prevents crashes if data variables are highly correlated or constant
    sim_log_inputs = np.random.multivariate_normal(mu_log.values, cov_log.values, N_SIMULATIONS, check_valid='warn', method='eigh')
except Exception as e:
    print(f"⚠️ Warning: Matrix singularity issue ({e}). Falling back to 'svd' method.")
    # Fallback method for tricky matrices
    sim_log_inputs = np.random.multivariate_normal(mu_log.values, cov_log.values, N_SIMULATIONS, check_valid='ignore', method='svd')

# C. Convert back to Real Scale
sim_inputs_df = pd.DataFrame(np.exp(sim_log_inputs), columns=found_vars)

# D. Run Regression
sim_pm25 = np.full(N_SIMULATIONS, betas['const'])
for col in found_vars:
    sim_pm25 += betas[col] * sim_inputs_df[col].values

# E. Inject Residual Noise
noise = np.random.normal(0, resid_std, N_SIMULATIONS)
sim_pm25 += noise

# F. Enforce Physics
sim_pm25 = np.maximum(sim_pm25, 0)

# ==============================================================================
# 5. RESULTS & VISUALIZATION
# ==============================================================================
mean_sim = np.mean(sim_pm25)
upper_bound = np.percentile(sim_pm25, 95)
prob_extreme = np.mean(sim_pm25 > 150) * 100

print(f"\n[Forecast Results]")
print(f"Mean: {mean_sim:.2f} | 95% Worst Case: {upper_bound:.2f}")
print(f"Risk > 150 (Hazardous): {prob_extreme:.2f}%")

# FIGURE 2: Forecast
plt.figure(figsize=(12, 7), num="Figure 2: Forecast Results")
sns.histplot(sim_pm25, bins=70, kde=True, stat="density", color="teal", alpha=0.4, label="Forecast")
plt.axvspan(35, 150, color='orange', alpha=0.05, label='Unhealthy')
plt.axvspan(150, max(sim_pm25.max(), 200), color='red', alpha=0.05, label='Hazardous')
plt.axvline(mean_sim, color='blue', linestyle='-', label=f'Mean ({mean_sim:.0f})')
plt.axvline(upper_bound, color='red', linestyle='--', label=f'95% Worst ({upper_bound:.0f})')
plt.title("Probabilistic PM2.5 Forecast (Correlated)", fontsize=16, weight='bold')
plt.xlabel("PM2.5 Concentration")
plt.legend()
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# FIGURE 3: Sensitivity
print("\n--- [Phase 4] Sensitivity Analysis ---")
correlations = {}
for col in found_vars:
    corr, _ = spearmanr(sim_inputs_df[col], sim_pm25)
    correlations[col] = corr

sens_df = pd.DataFrame(list(correlations.items()), columns=['Input', 'Correlation'])
sens_df = sens_df.sort_values(by='Correlation', key=abs, ascending=False)

plt.figure(figsize=(10, 6), num="Figure 3: Sensitivity Analysis")
sns.barplot(x='Correlation', y='Input', data=sens_df, palette='coolwarm')
plt.title("Sensitivity: Which variables drive risk?", fontsize=14, weight='bold')
plt.axvline(0, color='black')
plt.tight_layout()
plt.subplots_adjust(top=0.93)

print("\nDone! Showing all plots...")
plt.show()
