import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT YOUR MASTER LOADER
import data 

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set to True to analyze only the Haze Season (June-Sept)
FILTER_HAZE_SEASON = False 
N_SIMULATIONS = 10000

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("--- Loading Data ---")
df_clean = data.get_data()

# Optional: Seasonal Filtering
if FILTER_HAZE_SEASON:
    print("⚠️ FILTER ACTIVE: Using only Haze Season (June-Sept) data.")
    # Extract month from date
    df_clean['month'] = df_clean['date'].dt.month
    df_clean = df_clean[df_clean['month'].isin([6, 7, 8, 9])]
    print(f"Data filtered to {len(df_clean)} rows.")

# Define Model Variables
# Note: We take logs of inputs because pollutants follow Log-Normal distributions.
# This makes the math cleaner and handles the "Heavy Tail" automatically.
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 
input_vars = [col for col in input_vars if col in df_clean.columns]
target_var = 'pm25'

print(f"Using inputs: {input_vars}")

# ==============================================================================
# 2. DETERMINISTIC MODEL (OLS Baseline)
# ==============================================================================
print("\n--- [Phase 1] Training Deterministic Model ---")
X = df_clean[input_vars]
Y = df_clean[target_var]
X_const = sm.add_constant(X)

# Fit OLS
model = sm.OLS(Y, X_const).fit()
betas = model.params
resid_std = model.resid.std() # The "Unexplained Variance" (Noise)

print(model.summary())
print(f"\nModel Residual Std Dev (Noise): {resid_std:.2f}")

# ==============================================================================
# 3. MULTIVARIATE STOCHASTIC SIMULATION (The "Pro" Upgrade)
# ==============================================================================
print("\n--- [Phase 2] Correlated Monte Carlo Simulation ---")

# A. Prepare the Covariance Structure
# We assume inputs are Log-Normally distributed. 
# Therefore, their LOGS are Normally distributed.
# We work in "Log-Space" to use Cholesky Decomposition easily.
log_data = np.log(df_clean[input_vars].replace(0, 0.001)) # Avoid log(0)
mu_log = log_data.mean()
cov_log = log_data.cov()

# B. Generate Correlated Random Variables (The Cholesky Step)
# This replaces the independent loops. numpy.random.multivariate_normal 
# uses covariance to generate linked variables automatically.
print(f"Generating {N_SIMULATIONS} scenarios using Multivariate Log-Normal...")
sim_log_inputs = np.random.multivariate_normal(mu_log, cov_log, N_SIMULATIONS)

# C. Convert back to Real Scale (Exponentiate)
sim_inputs_df = pd.DataFrame(np.exp(sim_log_inputs), columns=input_vars)

# D. Run the Regression Equation
# PM2.5 = Intercept + (Beta1 * Sim_PM10) + ... + Noise
sim_pm25 = np.full(N_SIMULATIONS, betas['const'])

for col in input_vars:
    sim_pm25 += betas[col] * sim_inputs_df[col].values

# E. Inject Residual Noise (Model Uncertainty)
# This accounts for the fact that our model isn't perfect.
noise = np.random.normal(0, resid_std, N_SIMULATIONS)
sim_pm25 += noise

# F. Enforce Physics (No negative pollution)
sim_pm25 = np.maximum(sim_pm25, 0)

# ==============================================================================
# 4. RESULTS & ANALYSIS
# ==============================================================================
mean_sim = np.mean(sim_pm25)
upper_bound = np.percentile(sim_pm25, 95)
prob_unhealthy = np.mean(sim_pm25 > 55) * 100
prob_extreme = np.mean(sim_pm25 > 150) * 100

print(f"\n[Stochastic Forecast Results]")
print(f"Mean Prediction: {mean_sim:.2f}")
print(f"95% Worst Case:  {upper_bound:.2f}")
print(f"Risk > 55 (Unhealthy): {prob_unhealthy:.2f}%")
print(f"Risk > 150 (Hazardous): {prob_extreme:.2f}%")

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
# Set up layout
sns.set_theme(style="whitegrid", palette="muted")

# FIGURE 1: Forecast Distribution
plt.figure(figsize=(12, 6), num="Figure 1: Probabilistic Forecast")
sns.histplot(sim_pm25, bins=70, kde=True, stat="density", color="teal", alpha=0.4, label="Forecast Probability")

# Risk Zones
plt.axvspan(35, 150, color='orange', alpha=0.1, label='Unhealthy (35-150)')
plt.axvspan(150, max(sim_pm25.max(), 200), color='red', alpha=0.1, label='Hazardous (>150)')

plt.axvline(mean_sim, color='blue', linestyle='-', linewidth=2, label=f'Mean ({mean_sim:.0f})')
plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label=f'95% Risk ({upper_bound:.0f})')

plt.title("Probabilistic PM2.5 Forecast (Correlated Monte Carlo)", fontsize=14, weight='bold')
plt.xlabel("PM2.5 Concentration")
plt.legend()
plt.xlim(0, np.percentile(sim_pm25, 99)) # Crop extreme outliers for readability
plt.tight_layout()

# FIGURE 2: Sensitivity Analysis (Tornado)
print("\n--- [Phase 3] Sensitivity Analysis ---")
correlations = {}
for col in input_vars:
    # We correlate the SIMULATED inputs with the SIMULATED output
    corr, _ = spearmanr(sim_inputs_df[col], sim_pm25)
    correlations[col] = corr

sens_df = pd.DataFrame(list(correlations.items()), columns=['Input', 'Correlation'])
sens_df['Abs_Impact'] = sens_df['Correlation'].abs()
sens_df = sens_df.sort_values('Abs_Impact', ascending=False)

plt.figure(figsize=(10, 6), num="Figure 2: Sensitivity")
sns.barplot(x='Correlation', y='Input', data=sens_df, palette='coolwarm')
plt.title("Sensitivity: Which variables drive the haze?", fontsize=14, weight='bold')
plt.axvline(0, color='black')
plt.tight_layout()

print("\nDone! Showing plots...")
plt.show()
