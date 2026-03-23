"""
Psychophysiological Recovery Patterns after Alexander Technique-based Music Intervention
Data Analysis and Machine Learning (UMAP-GMM) Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap

# ---------------------------------------------------------
# 1. Data Loading and Preprocessing
# ---------------------------------------------------------
# Ensure 'response_github.xlsx' is in the same directory as this script.
file_path = "response_github.xlsx"
df = pd.read_excel(file_path)

# Correct typos and calculate BMI
df.rename(columns={'Hight': 'Height'}, inplace=True)
df['BMI'] = df['Weight'] / (df['Height']/100)**2

# Map Group labels (1: AT Intervention, 2: Non-AT Control)
df['Group'] = df['YN'].map({1: 'AT', 2: 'Non-AT'})

# Calculate Delta values (Post - Pre)
df['Delta_Cortisol'] = df['post_cortisol'] - df['pre_cortisol']
df['Delta_PHQ9'] = df['post_phq9'] - df['pre_phq9']
df['Delta_GAD7'] = df['post_gad7'] - df['pre_gad7']

# Handle missing values using Mean Imputation for the Delta variables
target_cols = ['Delta_Cortisol', 'Delta_PHQ9', 'Delta_GAD7']
df_imputed = df.copy()
for col in target_cols:
    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

# ---------------------------------------------------------
# 2. Baseline Characteristics (Table 2)
# ---------------------------------------------------------
print("\n[ Table 2: Baseline demographic and physical characteristics ]")
baseline_vars = {'Age': 'Age (years)', 'Height': 'Height (cm)', 'Weight': 'Weight (kg)', 'BMI': 'BMI (kg/m²)'}
results_baseline = []

for var, label in baseline_vars.items():
    at = df[df['Group'] == 'AT'][var].dropna()
    non = df[df['Group'] == 'Non-AT'][var].dropna()
    t_stat, p_val = ttest_ind(at, non, equal_var=False)
    results_baseline.append([label, f"{non.mean():.2f} +/- {non.std():.2f}", f"{at.mean():.2f} +/- {at.std():.2f}", f"{p_val:.3f}"])

df_baseline = pd.DataFrame(results_baseline, columns=['Variable', 'Non-AT group', 'AT Intervention', 'p-value'])
print(df_baseline.to_string(index=False))

# ---------------------------------------------------------
# 3. Within-group Pre-Post Comparison (Table 3)
# ---------------------------------------------------------
print("\n[ Table 3: Within-group comparison of pre- and post-intervention scores ]")
compare_vars_rel = {'Cortisol': ('pre_cortisol', 'post_cortisol'), 'PHQ-9': ('pre_phq9', 'post_phq9'), 'GAD-7': ('pre_gad7', 'post_gad7')}
paired_results = []

for group in ['AT', 'Non-AT']:
    df_g = df[df['Group'] == group]
    for var_name, (pre, post) in compare_vars_rel.items():
        valid = df_g[[pre, post]].dropna()
        t_stat, p_val = ttest_rel(valid[pre], valid[post])
        paired_results.append({'Group': group, 'Variable': var_name, 
                               'Pre (Mean+/-SD)': f"{valid[pre].mean():.3f} +/- {valid[pre].std():.3f}",
                               'Post (Mean+/-SD)': f"{valid[post].mean():.3f} +/- {valid[post].std():.3f}",
                               'p-value': f"{p_val:.3f}"})

df_paired = pd.DataFrame(paired_results)
print(df_paired.to_string(index=False))

# ---------------------------------------------------------
# 4. Between-group Comparison of Delta Values (Table 4)
# ---------------------------------------------------------
print("\n[ Table 4: Between-group comparison of mean Delta values ]")
ind_results = []
for var in target_cols:
    at_g = df_imputed[df_imputed['Group'] == 'AT'][var].dropna()
    non_g = df_imputed[df_imputed['Group'] == 'Non-AT'][var].dropna()
    t_stat, p_val = ttest_ind(at_g, non_g, equal_var=False)
    ind_results.append({'Variable': var, 
                        'AT group (Mean+/-SD)': f"{at_g.mean():.3f} +/- {at_g.std():.3f}",
                        'Non-AT group (Mean+/-SD)': f"{non_g.mean():.3f} +/- {non_g.std():.3f}",
                        'p-value': f"{p_val:.3f}"})

df_ind = pd.DataFrame(ind_results)
print(df_ind.to_string(index=False))

# ---------------------------------------------------------
# 5. UMAP Dimensionality Reduction & GMM Clustering
# ---------------------------------------------------------
X = df_imputed[target_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP (3D)
umap_model = umap.UMAP(n_components=3, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# GMM Clustering (k=3)
gmm_final = GaussianMixture(n_components=3, random_state=42)
labels_final = gmm_final.fit_predict(X_umap)
df_imputed['UMAP_GMM'] = labels_final

# ---------------------------------------------------------
# 6. Cluster-level Comparison (ANOVA & Tukey HSD) (Table 5)
# ---------------------------------------------------------
print("\n[ Table 5: Cluster-level comparison (ANOVA & Tukey HSD) ]")
for var in target_cols:
    groups = [df_imputed[df_imputed['UMAP_GMM'] == g][var].dropna() for g in sorted(df_imputed['UMAP_GMM'].unique())]
    f_stat, p_val = f_oneway(*groups)
    print(f"\nVariable: {var} | ANOVA F = {f_stat:.3f}, p = {p_val:.4f}")
    
    if p_val < 0.05:
        valid_data = df_imputed[['UMAP_GMM', var]].dropna()
        tukey = pairwise_tukeyhsd(endog=valid_data[var], groups=valid_data['UMAP_GMM'], alpha=0.05)
        print(tukey)

# ---------------------------------------------------------
# 7. Visualizations (High-Resolution Output)
# ---------------------------------------------------------
# Figure 2: 3D UMAP-GMM Clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=labels_final, s=60, cmap='Set1')
ax.set_title("UMAP-GMM Clustering (k=3)", fontsize=14)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_zlabel("UMAP 3", fontsize=12)

legend_labels = [f"Cluster {i}" for i in range(3)]
handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10) for i in range(3)]
ax.legend(handles=handles, title='Clusters', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.savefig("Figure2_UMAP_GMM_3D.jpg", dpi=300, format='jpg', bbox_inches='tight')
plt.close()

# Figure 3: Latent Space Mapping (2D Spider Plot)
X_2d = X_umap[:, :2]
centroids = np.array([X_2d[labels_final == i].mean(axis=0) for i in np.unique(labels_final)])
cluster_colors = {0: '#5DA5DA', 1: '#FAA43A', 2: '#60BD68'}
group_markers = {'AT': 'o', 'Non-AT': 'x'}

plt.figure(figsize=(9, 8))
for i in np.unique(labels_final):
    for g in np.unique(df_imputed['Group']):
        mask = (labels_final == i) & (df_imputed['Group'] == g)
        if mask.any():
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=cluster_colors[i], label=f'Cluster {i} ({g})',
                        marker=group_markers[g], edgecolor='k' if group_markers[g] == 'o' else None, s=80, alpha=0.8)

for i, c in enumerate(centroids):
    cluster_points = X_2d[labels_final == i]
    for p in cluster_points:
        plt.plot([p[0], c[0]], [p[1], c[1]], color='gray', linewidth=0.5, alpha=0.4)
    plt.scatter(c[0], c[1], s=300, color='black', marker='*')
    plt.text(c[0]+0.1, c[1], f'P{i}', fontsize=13, weight='bold', va='center')

plt.xlabel("Latent Dimension 1", fontsize=12)
plt.ylabel("Latent Dimension 2", fontsize=12)
plt.title("Latent Space Mapping by Cluster and Group", fontsize=14, pad=10)
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)

plt.savefig("Figure3_Latent_Space_Mapping.jpg", dpi=300, format='jpg', bbox_inches='tight')
plt.close()

print("\nAnalysis complete. High-resolution figures have been saved to the directory.")
