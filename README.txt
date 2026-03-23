Psychophysiological Recovery Patterns in Adolescents after Alexander Technique-based Music Intervention

This repository contains the dataset and Python analysis code used in the study exploring the effects of an Alexander Technique (AT)-based music intervention on the psychological and physiological recovery of adolescents. The study utilizes a hybrid unsupervised learning approach, combining Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction with a Gaussian Mixture Model (GMM) for clustering, to identify latent recovery trajectories based on physiological (Cortisol) and psychological (PHQ-9, GAD-7) stress markers.

Repository Contents
• response_github.xlsx: A de-identified dataset provided for demonstration and reproducibility purposes.
• analysis_script.py: The complete Python script used for statistical analysis, machine learning clustering, and high-resolution figure generation.
• Figures: Running the script will automatically generate 300 DPI high-resolution JPG files for the 3D UMAP-GMM Clustering (Figure 2) and the Latent Space Mapping (Figure 3).

Requirements
The analysis was performed in a Python 3.13 environment. The following libraries are required to run the script successfully:
• pandas
• numpy
• scipy
• statsmodels
• scikit-learn
• umap-learn
• matplotlib
• seaborn
• openpyxl

You can install these dependencies using pip:
pip install pandas numpy scipy statsmodels scikit-learn umap-learn matplotlib seaborn openpyxl

How to Run
1. Download or clone this repository to your local machine.
2. Ensure both analysis_script.py and response_github.xlsx are located in the same directory.
3. Execute the Python script.
4. The console will output the statistical results corresponding to Table 2 (Baseline characteristics), Table 3 (Within-group comparison), Table 4 (Between-group comparison), and Table 5 (Cluster-level ANOVA and Tukey HSD) of the manuscript.
5. High-resolution images for the manuscript figures will be saved automatically in the working directory.
Data Processing Details
• Variables: Changes in stress markers (Delta_Cortisol, Delta_PHQ-9, Delta_GAD-7) were calculated by subtracting pre-intervention values from post-intervention values.
• Missing Values: Missing data in the $\Delta$ variables were handled using mean imputation to maintain statistical power prior to the clust ring process. The logic is explicitly included in the provided script.
• Clustering Architecture: Data was standardized, reduced to 3 dimensions via UMAP, and clustered using GMM (k=3).

Ethical Approval and Data Availability
This study protocol was reviewed and approved by the Public Institutional Bioethics Committee under the Ministry of Health and Welfare of the Republic of Korea (IRB No. P01-202308-06-002). The original raw datasets contain sensitive human participant information and cannot be made fully public due to institutional data and bioethics protection policies. Access to the original de-identified data may be granted by the corresponding author upon reasonable request.
