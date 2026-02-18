### Breast Cancer Prediction Using Machine Learning

**Author**: Aakar Shah

#### Executive Summary

This project develops a machine learning classification model to predict breast cancer malignancy (malignant vs. benign) using cellular measurements from fine needle aspirate (FNA) samples. Using the Breast Cancer Wisconsin (Diagnostic) dataset containing 569 patient observations with 30 numerical features, I performed comprehensive exploratory data analysis, feature engineering, and built a baseline logistic regression model achieving over 95% accuracy. The analysis identified key cellular characteristics—such as concave points, area, and radius—as the strongest predictors of malignancy. This work demonstrates how machine learning can provide objective, data-driven support for clinical breast cancer diagnosis.

#### Rationale

Breast cancer is the most common cancer among women worldwide, with approximately 2.3 million new cases diagnosed annually. Early and accurate diagnosis is critical—detecting cancer at Stage 1 provides a 99% five-year survival rate, compared to 28% at Stage 4. Traditional diagnostic methods rely heavily on pathologist expertise, which can be subjective and time-consuming. 

This project addresses the need for automated, objective diagnostic support tools that can:
- Provide consistent, standardized assessments across different medical centers
- Reduce pathologist workload and potential diagnostic errors
- Identify which cellular measurements are most critical for diagnosis
- Support faster diagnosis, enabling earlier treatment interventions
- Reduce unnecessary follow-up procedures by improving initial assessment accuracy

#### Research Question

Can machine learning models accurately predict breast cancer malignancy from cellular measurements of fine needle aspirate samples, and which characteristics are most predictive of malignancy?

#### Data Sources

**Primary Data Source**: Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository  
**Link**: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Dataset Characteristics**:
- 569 patient observations (357 benign, 212 malignant)
- 30 continuous numerical features computed from digitized FNA images
- Features measure 10 cellular characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- Each characteristic measured as: mean, standard error, and worst (largest) value
- Binary target variable: Malignant (M) or Benign (B)
- No missing values
- Pre-cleaned and ready for analysis

#### Methodology

**1. Exploratory Data Analysis (EDA)**
- Examined data distribution, summary statistics, and data types
- Analyzed class balance (62.7% benign, 37.3% malignant)
- Created correlation heatmap to identify multicollinearity
- Visualized feature distributions using histograms and box plots
- Generated pairplots to examine feature relationships

**2. Data Cleaning**
- Verified no missing values present
- Checked for and removed duplicate records (none found)
- Identified outliers using box plots and IQR method
- Decided to retain outliers as they may represent legitimate extreme cases

**3. Feature Engineering**
- Applied StandardScaler to normalize all 30 features
- Split data into training (80%) and testing (20%) sets with stratification
- Preserved class distribution in both train and test sets

**4. Baseline Model Development**
- Built Logistic Regression as baseline classification model
- Chose logistic regression for interpretability and as a standard baseline
- Trained on scaled training data
- Evaluated on held-out test set

**5. Model Evaluation**
- Used multiple metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Created confusion matrix to analyze misclassifications
- Generated classification report for detailed performance metrics
- Prioritized recall (sensitivity) to minimize false negatives in cancer detection

#### Results

**Model Performance**:
- **Accuracy**: 96.5% on test set
- **Precision**: 95.2% (few false positives)
- **Recall**: 97.7% (very few false negatives - critical for cancer detection)
- **F1-Score**: 96.4% (balanced performance)
- **ROC-AUC**: 0.99 (excellent discrimination ability)

**Key Findings**:

1. **Most Predictive Features**: 
   - Worst concave points (strongest predictor)
   - Worst perimeter and worst area
   - Mean concave points and mean concavity
   - These "worst" features (largest measured values) consistently showed strongest correlations with malignancy

2. **Feature Correlations**:
   - High multicollinearity among radius, perimeter, and area measurements (correlation > 0.95)
   - Texture features showed lower correlation with size-based features
   - "Worst" measurements generally more predictive than "mean" measurements

3. **Class Distribution**:
   - Dataset is slightly imbalanced: 62.7% benign, 37.3% malignant
   - Sufficient representation of both classes for reliable model training

4. **Model Insights**:
   - Logistic regression baseline performs exceptionally well (>95% accuracy)
   - High recall (97.7%) is critical - model rarely misses malignant cases
   - Few false negatives minimize risk of missing cancer diagnosis
   - Model is interpretable, allowing clinicians to understand predictions

**Clinical Implications**:
- The model could serve as an effective screening tool to flag high-risk cases
- High recall ensures minimal false negatives (missed cancers)
- Focus on "worst" cellular measurements aligns with pathology practice of examining most abnormal cells

#### Next Steps

**For Module 24 - Final Capstone Submission**:

1. **Advanced Modeling**:
   - Implement Decision Trees and k-Nearest Neighbors for comparison
   - Apply GridSearchCV for hyperparameter tuning
   - Try Ridge/Lasso regularization to address multicollinearity
   - Implement Principal Component Analysis (PCA) for dimensionality reduction

2. **Feature Engineering Enhancements**:
   - Create polynomial features to capture non-linear relationships
   - Experiment with feature selection techniques
   - Address multicollinearity more systematically

3. **Model Interpretation**:
   - Extract and visualize feature importance/coefficients
   - Create ROC curves comparing multiple models
   - Develop confusion matrices for all models
   - Analyze which features drive predictions

4. **Documentation & Presentation**:
   - Clean and organize code for readability
   - Create comprehensive visualizations for non-technical audience
   - Develop executive summary with business recommendations
   - Prepare presentation materials explaining findings to healthcare stakeholders

5. **Validation**:
   - Implement k-fold cross-validation for robust performance estimates
   - Test model sensitivity to different train/test splits
   - Analyze error cases to understand model limitations

#### Outline of Project

- [Exploratory Data Analysis Notebook](./exploratory_analysis.ipynb)

##### Contact and Further Information

**GitHub Repository**: https://github.com/aashah2022/breast-cancer-prediction  
**Date:** February 2026  
**Course:** Module 20 - Capstone EDA
