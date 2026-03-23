# Breast Cancer Prediction Using Machine Learning

**Author:** Aakar Shah  
**Date:** March 2026  
**Course:** Final Capstone

---

## Executive Summary

This project uses machine learning to predict whether breast tumors are malignant (cancerous) or benign (non-cancerous) based on measurements from fine needle aspirate (FNA) samples. I analyzed the Breast Cancer Wisconsin dataset with 569 patient samples and 30 features, then built and optimized several classification models.

**Main Results:**
- Final model achieved **98% accuracy** on test data
- **Zero false positives** - didn't incorrectly flag any benign tumors as malignant
- Only **2 false negatives** - missed 2 cancers out of 42 (95% recall)
- **ROC-AUC of 0.998** - nearly perfect at distinguishing between classes
- Most important features: worst texture, worst radius, and worst area

The model could help doctors make faster, more consistent diagnoses and reduce unnecessary follow-up procedures.

---

## Rationale

Breast cancer is one of the most common cancers affecting women worldwide. Early detection is critical - catching cancer at Stage 1 gives patients a 99% five-year survival rate, compared to only 28% at Stage 4. However, current diagnostic methods have some challenges:

- Diagnosis relies heavily on individual pathologist expertise, which can vary
- Manual analysis of samples takes hours or days
- Pathologists are dealing with increasing workloads
- Rural areas may not have access to specialist pathologists
- Diagnostic uncertainty can lead to unnecessary biopsies (which cost $1,000-$5,000 each)

Machine learning models could address these issues by providing:
- Fast, consistent analysis (results in seconds)
- A standardized "second opinion" for doctors
- Reduced unnecessary procedures by improving accuracy
- Better access to expert-level screening in underserved areas

This project explores whether ML can accurately predict breast cancer malignancy and identifies which cellular features are most important for diagnosis.

---

## Research Question

**Can machine learning models accurately predict breast cancer malignancy from cellular measurements of fine needle aspirate samples, and which characteristics are most predictive of malignancy?**

---

## Data Sources

I used the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository:
- **Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Source:** University of Wisconsin Hospitals, Madison (collected in 1990s)

**Dataset Details:**
- 569 patients total (357 benign, 212 malignant)
- 30 numerical features describing cell nuclei characteristics
- Features include: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- Each characteristic measured three ways: mean, standard error, and "worst" (largest value)
- Target variable: Malignant (M) or Benign (B)
- No missing values - the dataset was already clean

The data comes from digitized images of FNA samples. Doctors use a fine needle to extract cells from breast masses, then these cells are analyzed under a microscope.

---

## Methodology

### 1. Exploratory Data Analysis

First, I explored the data to understand its characteristics:

**Data Quality Checks:**
- Verified no missing values or duplicate rows
- Checked class distribution: 62.7% benign, 37.3% malignant
- Looked at feature distributions - most were right-skewed
- Found some outliers but decided to keep them since they might represent real extreme cases

**Feature Analysis:**
- Created a correlation heatmap showing relationships between features
- Found high correlations (>0.9) between related features like radius, perimeter, and area
- Compared benign vs malignant tumors using box plots
- Noticed that malignant tumors generally have higher values for most measurements

**Key Insights from EDA:**
- Size-related features (radius, perimeter, area) show clear differences between benign and malignant
- "Worst" measurements (the largest values) seem to be more important than mean values
- High correlation between some features suggests I should use regularization or PCA

### 2. Data Preprocessing

**Prepared the data for modeling:**
- Encoded target variable: Malignant = 1, Benign = 0
- Split data: 80% training (455 samples), 20% testing (114 samples)
- Used stratification to maintain the same class proportions in both sets
- Applied StandardScaler to normalize features (mean=0, standard deviation=1)

This scaling is important because features have very different ranges - for example, area ranges from 143 to 2501 while fractal dimension ranges from 0.05 to 0.09.

### 3. Model Comparison

I built and compared three different classification models:

**Model 1: Logistic Regression**
- Simple linear classifier that's easy to interpret
- Good baseline model
- Achieved 96.5% accuracy on test set
- Fast to train and make predictions

**Model 2: Decision Tree**
- Can capture non-linear patterns in data
- Very interpretable - you can visualize the decision rules
- Achieved 92.1% accuracy on test set
- Lowest performance of the three models

**Model 3: k-Nearest Neighbors (k-NN)**
- Classifies based on similarity to nearby data points
- Simple concept, easy to understand
- Achieved 95.6% accuracy on test set
- Slower for predictions on large datasets

**Evaluation:**
- Used 5-fold cross-validation to get robust performance estimates
- Compared models using accuracy, precision, recall, F1-score, and ROC-AUC
- Created confusion matrices to see what types of errors each model made
- Plotted ROC curves to visualize performance

**Result:** Logistic Regression performed best overall and was selected for optimization.

### 4. PCA Experiment

I tested Principal Component Analysis (PCA) to reduce features from 30 to 10:
- PCA retained over 95% of the variance in the data
- Interestingly, Logistic Regression actually improved to 97.4% accuracy with PCA
- Decision Tree performance also increased significantly with PCA
- This suggests we could potentially use a simpler model with fewer features

### 5. Model Optimization

**Used GridSearchCV to find the best hyperparameters:**
- Tested different values of C (regularization strength): [0.001, 0.01, 0.1, 1, 10, 100]
- Tested both L1 (Lasso) and L2 (Ridge) regularization
- Used 5-fold cross-validation within the grid search
- Let the algorithm test all combinations to find the best settings

**Regularization Comparison:**
- Compared Ridge (L2) and Lasso (L1) regularization side-by-side
- Both performed well, showing the model is robust to regularization choice

**Final Result:** The optimized model achieved **98% accuracy** on the test set.

### 6. Feature Importance

Analyzed which features the model found most important:

**Top 5 Most Predictive Features:**
1. texture3 (worst texture) - most important
2. radius3 (worst radius)
3. area3 (worst area)  
4. radius2 (SE radius)
5. perimeter3 (worst perimeter)

**What this means clinically:**
- The "worst" measurements (largest observed values) are the best predictors
- Texture irregularity is the single strongest indicator of malignancy
- Cell size features (radius, area, perimeter) are collectively very important
- These findings make sense and align with what pathologists already know

---

## Results

### Final Model Performance

**Test Set Results (114 samples):**

| Metric | Benign | Malignant | Overall |
|--------|--------|-----------|---------|
| Precision | 0.97 | 1.00 | 0.98 |
| Recall | 1.00 | 0.95 | 0.98 |
| F1-Score | 0.99 | 0.98 | 0.98 |

**Overall Metrics:**
- **Accuracy:** 98% (112 correct out of 114 cases)
- **ROC-AUC:** 0.998 (almost perfect)

### Confusion Matrix

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actually Benign** | 72 ✓ | 0 |
| **Actually Malignant** | 2 | 40 ✓ |

**What this means:**
- **72 benign tumors:** All correctly identified (no false positives!)
- **42 malignant tumors:** 40 correctly identified, 2 missed
- **Zero false positives:** No benign tumors were incorrectly flagged as malignant
- **2 false negatives:** Only 2 cancers were missed (4.8% miss rate)

### Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression (baseline) | 96.5% | 0.96 | 0.98 | 0.99 |
| Decision Tree | 92.1% | 0.92 | 0.93 | 0.94 |
| k-NN | 95.6% | 0.96 | 0.95 | 0.98 |
| **Optimized Logistic Regression** | **98.0%** | **0.98** | **0.98** | **0.998** |

The optimized logistic regression clearly outperformed the other models.

### Key Findings

**What worked well:**
1. GridSearchCV hyperparameter tuning improved accuracy from 96.5% to 98%
2. Zero false positives is excellent - no unnecessary anxiety for patients
3. 95% recall for malignant cases means we catch almost all cancers
4. When the model predicts malignant, it's 100% correct (perfect precision)
5. The model's predictions align with what doctors already look for

**What I learned:**
1. "Worst" measurements are more predictive than mean values
2. Texture irregularity is the strongest single predictor
3. Simple models (logistic regression) can work very well with proper optimization
4. PCA can actually improve performance while reducing complexity
5. The high correlation between size features confirms the need for regularization

---

## Business Impact

### Why This Matters

**For Patients:**
- Faster diagnosis (seconds vs hours/days) means less waiting and anxiety
- Zero false positives means no unnecessary worry about benign tumors
- 95% of cancers detected automatically enables earlier treatment
- Earlier treatment means better survival rates

**For Doctors and Hospitals:**
- Reduces pathologist workload - they can focus on difficult cases
- Provides a consistent "second opinion" for every case
- Standardizes diagnosis quality across all cases
- Could save time equivalent to 2-3 hours per pathologist per day

**Cost Savings:**
For a hospital processing 1,000 FNA samples per year:
- Current false positive rate: ~3-5% → 30-50 unnecessary biopsies
- Cost per biopsy: $1,000-$5,000
- **Current waste:** $30,000-$250,000 per year on unnecessary procedures
- **With this model (0% false positives):** Potential to eliminate this waste entirely

### Important Limitations

**This model should NOT be used alone for diagnosis because:**
1. It missed 2 out of 42 cancers (4.8% miss rate)
2. It was trained on 1990s data - needs validation on modern equipment
3. It only does binary classification (malignant vs benign) - can't predict cancer stage
4. It needs to be integrated properly into clinical workflows
5. Doctors must always review and approve the predictions

**The model should be used as:**
- A screening tool to help prioritize cases
- A "second opinion" that doctors can reference
- A consistency check to catch potential errors
- A teaching tool for less experienced pathologists

---

## Limitations

### Data Limitations
- Dataset is from the 1990s - imaging technology has improved since then
- Only 569 samples - more data would make the model more robust
- All data from one hospital - may not generalize to other populations or equipment
- No information about patient demographics or medical history

### Model Limitations
- Can only classify as malignant or benign (no information about cancer type, stage, or severity)
- Requires exactly 30 specific measurements from FNA images
- Missed 2 malignant cases - needs 100% human oversight
- No temporal information - can't predict progression or recurrence

### Implementation Challenges
- Would need FDA approval before clinical use
- Requires integration with hospital IT systems
- Doctors and technicians need training on how to use it
- Legal liability questions around AI-assisted diagnosis
- Must be validated on current imaging technology and diverse patient populations

---

## Next Steps

### What Should Happen Next

**Short-term (3-6 months):**
1. **Run a pilot study** with 100-200 recent cases
   - Compare model predictions to actual diagnoses
   - Measure time savings and accuracy
   - Get feedback from pathologists on usability

2. **Validate on modern data**
   - Test with current FNA imaging technology
   - Include diverse patient populations
   - Try different imaging equipment brands

3. **Clinical integration planning**
   - Figure out how to fit this into existing workflows
   - Design the user interface for pathologists
   - Create protocols for handling disagreements between model and doctor

**Long-term:**
1. **Try more advanced models**
   - Ensemble methods (Random Forest, XGBoost)
   - Deep learning on raw images (instead of extracted features)
   - Expected improvement: 98% → 99%+

2. **Expand capabilities**
   - Multi-class classification (predict cancer subtype)
   - Predict cancer stage or aggressiveness
   - Incorporate patient history and other clinical data

3. **Collect more data**
   - Partner with multiple hospitals for larger dataset
   - Include longitudinal data (track patients over time)
   - Collect data on treatment outcomes

---

## How to Run This Project

### Prerequisites
```bash
Python 3.8 or higher
Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, ucimlrepo
```

### Installation
```bash
# Clone the repository
git clone https://github.com/aashah2022/breast-cancer-prediction.git
cd breast-cancer-prediction

# Install packages
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo

# Open Jupyter
jupyter notebook
```

### Running the Notebooks
Run the notebooks in this order:
1. `01_exploratory_data_analysis.ipynb` - Data exploration and preprocessing
2. `02_model_comparison.ipynb` - Build and compare multiple models
3. `03_final_model_optimization.ipynb` - Optimize best model and generate final results

The data is automatically downloaded from UCI when you run the first notebook - no manual download needed.

---

## Project Structure
```
breast-cancer-prediction/
├── README.md
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_final_model_optimization.ipynb
```

---

## Conclusion

This project shows that machine learning can be very effective for breast cancer diagnosis support. The final model achieved 98% accuracy with zero false positives and only 2 false negatives out of 114 test cases. 

**Key takeaways:**
- Logistic regression performed best after optimization (98% accuracy)
- "Worst" measurements (texture, radius, area) are the most predictive features
- The model could help doctors work faster and more consistently
- Zero false positives could save hospitals thousands of dollars in unnecessary procedures
- However, the model should only be used as a decision support tool, not as a replacement for doctors

**The bottom line:** Machine learning shows promise for assisting with breast cancer diagnosis, but it needs proper validation, integration, and human oversight before clinical deployment. With the right implementation, this type of tool could help detect cancer earlier and save lives.

---

## Contact

**Author:** Aakar Shah  
**GitHub:** https://github.com/aashah2022/breast-cancer-prediction  
**Project:** Final Capstone 
**Date:** March 2026

---

**Data Source:**  
Breast Cancer Wisconsin (Diagnostic) Dataset  
Dr. William H. Wolberg, University of Wisconsin Hospitals  
UCI Machine Learning Repository
