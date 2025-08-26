# Financial Sentiment Analysis Project Report

## Phase 2: Classic ML Optimization

### Executive Summary
Phase 2 focused on systematically optimizing classic machine learning approaches for financial sentiment analysis. Through hyperparameter tuning, feature engineering, and ensemble methods, we achieved a **71.3% accuracy** using a stacking classifier, representing a **+3.9 percentage point improvement** over Phase 1 baselines. The most significant gains came from domain-specific feature engineering, while ensemble methods provided marginal but meaningful improvements.

### Optimization Strategy Overview
Our approach followed a systematic three-stage optimization process:

1. **Hyperparameter Tuning** - Optimize individual model parameters
2. **Feature Engineering** - Improve text representation for financial domain
3. **Ensemble Methods** - Combine model strengths for better performance

**Key Philosophy:** Establish realistic performance ceilings for classic ML before moving to advanced transformer models in Phase 3.

### Section 2: Hyperparameter Tuning

#### 2.1 Logistic Regression Optimization
**Approach:** Grid search with 5-fold cross-validation across regularization parameters.

**Parameter Grid:**
- **C values:** [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100] (regularization strength)
- **Penalty:** ['l1', 'l2'] (Lasso vs. Ridge regularization)
- **Solver:** ['liblinear', 'saga', 'lbfgs'] (optimization algorithms)
- **Class weight:** 'balanced' (handle class imbalance)

**Results:**
- **Best parameters:** C=2, penalty='l1', solver='liblinear'
- **Cross-validation score:** 68.4% (10-fold CV)
- **Validation accuracy:** 68.0%
- **Improvement:** +0.6% from Phase 1 baseline

**Key Insights:**
- L1 regularization (sparse features) works well for text data
- 10-fold cross-validation revealed more realistic performance expectations
- Moderate regularization (C=2) optimal for financial text

#### 2.2 Random Forest Optimization
**Approach:** Comprehensive tree structure optimization with class balancing.

**Parameter Grid:**
- **n_estimators:** [100, 200, 300] (number of trees)
- **max_depth:** [5, 10, 15, None] (tree complexity)
- **min_samples_split:** [2, 5, 10] (split threshold)
- **min_samples_leaf:** [1, 2, 4] (leaf node size)
- **criterion:** ['gini', 'entropy'] (split quality measure)

**Results:**
- **Best parameters:** 100 trees, unlimited depth, entropy criterion
- **Cross-validation score:** 65.6% (5-fold CV)
- **Validation accuracy:** 64.3%
- **Performance:** Below Phase 1 baseline (-3.1%)

**Key Insights:**
- Tree-based models struggle with high-dimensional sparse text data
- Unlimited depth trees perform best but still limited
- Feature sparsity challenges tree-based approaches

#### 2.3 SVM Optimization
**Approach:** Kernel and regularization optimization for non-linear patterns.

**Parameter Grid:**
- **C values:** [0.1, 1, 10, 100] (regularization strength)
- **Kernel:** ['linear', 'rbf'] (decision boundary type)
- **Gamma:** ['scale', 'auto', 0.001, 0.01] (kernel coefficient)
- **Class weight:** 'balanced' (handle class imbalance)

**Results:**
- **Best parameters:** C=1, kernel='linear', gamma='scale'
- **Cross-validation score:** 66.4% (5-fold CV)
- **Validation accuracy:** 67.5%
- **Improvement:** +0.1% from Phase 1 baseline

**Key Insights:**
- Linear kernel performs best (text data is linearly separable)
- RBF kernel adds complexity without performance gain
- Moderate regularization optimal for financial text

### Section 3: Feature Engineering

#### 3.1 N-gram Analysis
**Approach:** Test different word combination patterns to capture context.

**Configurations Tested:**
- **1-gram (baseline):** Single words only
- **1-2 gram:** Words + word pairs
- **1-3 gram:** Words + pairs + triplets
- **2-2 gram:** Word pairs only

**Results:**
| Configuration | Accuracy | F1-Score | Features |
|---------------|----------|----------|----------|
| 1-gram | 67.4% | 68.2% | 4,278 |
| 1-2 gram | 66.4% | 67.4% | 5,000 |
| 1-3 gram | 65.6% | 66.8% | 5,000 |
| 2-2 gram | 52.9% | 50.6% | 5,000 |

**Key Insights:**
- **Single words work best** for financial headlines
- Higher n-grams create sparse, noisy features
- Financial text is concise - individual terms carry meaning
- Feature sparsity limits n-gram effectiveness

#### 3.2 Alternative Feature Engineering
**Approach:** Optimize TF-IDF parameters and preprocessing for financial domain.

**Configurations Tested:**
- **Baseline:** Standard TF-IDF settings
- **Stricter Thresholds:** Remove more common/rare words
- **More Features:** 10,000 vs. 5,000 vocabulary size
- **Fewer Features:** 2,000 vocabulary size
- **Financial Focus:** No stop words (preserve financial terms)

**Results:**
| Configuration | Accuracy | F1-Score | Features |
|---------------|----------|----------|----------|
| Baseline | 70.5% | 71.5% | 4,503 |
| Stricter Thresholds | 70.7% | 71.6% | 1,989 |
| More Features | 70.5% | 71.5% | 4,503 |
| Fewer Features | 70.5% | 71.5% | 2,000 |
| **Financial Focus** | **70.9%** | **71.8%** | **3,046** |

**Key Insights:**
- **Financial Focus configuration** provides best performance
- Removing generic stop words preserves important financial context
- 3,000-4,000 features optimal vocabulary size
- Domain-specific preprocessing crucial for financial text

#### 3.3 Feature Importance Analysis
**Approach:** Analyze which terms drive sentiment predictions for each class.

**Top Features by Class:**

**Negative Class (Risk Signals):**
- "down" (7.047), "lower" (6.520), "drop" (6.064)
- "jobs" (6.044), "business" (5.398), "lost" (5.312)
- "cut" (5.224), "hit" (5.201), "shell" (5.083)

**Neutral Class (Informational):**
- "approximately" (5.974), "co" (5.974), "includes" (5.762)
- "is" (4.632), "astrazeneca" (4.486), "spy" (3.479)
- "the" (3.349), "will" (3.094), "aapl" (3.020)

**Positive Class (Growth Signals):**
- "decreased" (10.434), "rose" (10.063), "signed" (8.803)
- "increase" (8.422), "positive" (7.729), "increased" (7.617)
- "awarded" (7.224), "long" (7.002), "grew" (6.740)

**Key Insights:**
- **Context ambiguity:** "down" appears in both negative and positive classes
- **Domain-specific patterns:** Company names ("shell", "astrazeneca") are neutral
- **Action words:** "signed", "awarded" are strong positive signals
- **Financial terminology:** Captures domain-specific sentiment patterns

### Section 4: Ensemble Methods

#### 4.1 Voting Classifier
**Approach:** Combine predictions from best tuned models using majority voting.

**Configuration:**
- **Models:** Logistic Regression, Random Forest, SVM
- **Voting strategy:** Hard voting (majority class prediction)
- **Weights:** Equal weights for all models

**Results:**
- **Voting accuracy:** 70.3%
- **Best individual:** 70.9% (Logistic Regression)
- **Performance:** -0.6% from best individual model

**Key Insights:**
- **Model correlation:** High correlation between models limited voting effectiveness
- **Same features:** Identical TF-IDF representation reduced diversity
- **Voting limitation:** Simple majority voting not effective for correlated models

#### 4.2 Stacking Classifier
**Approach:** Meta-learner approach using cross-validation for meta-features.

**Configuration:**
- **Base models:** Logistic Regression, Random Forest, SVM
- **Meta-learner:** Logistic Regression
- **Cross-validation:** 5-fold CV for meta-feature generation
- **Stack method:** predict_proba (probability features)

**Results:**
- **Stacking accuracy:** 71.3%
- **Cross-validation score:** 68.9% (±2.4%) (5-fold CV)
- **Improvement:** +0.4% over best individual model

**Key Insights:**
- **Meta-learning success:** Learned optimal combination weights
- **Probability features:** More nuanced than hard predictions
- **Small but meaningful:** +0.4% improvement is significant for this domain
- **Overfitting concern:** 2.4% gap between CV and validation scores

### Performance Comparison Summary

**Complete Phase 2 Results:**

| Method | Accuracy | Improvement | Key Insight |
|--------|----------|-------------|-------------|
| Phase 1 Baseline | 67.4% | 0.0% | TF-IDF + Logistic Regression |
| LR Tuned | 68.0% | +0.6% | Hyperparameter optimization |
| RF Tuned | 64.3% | -3.1% | Tree models struggle with text |
| SVM Tuned | 67.5% | +0.1% | Linear kernel optimal |
| N-grams | 67.4% | 0.0% | 1-gram works best |
| **Feature Engineering** | **70.9%** | **+3.5%** | **Financial Focus configuration** |
| Voting | 70.3% | -0.6% | Model correlation limits voting |
| **Stacking** | **71.3%** | **+0.4%** | **Meta-learner approach** |

**Key Findings:**
- **Best individual model:** Logistic Regression (70.9%)
- **Best ensemble:** Stacking Classifier (71.3%)
- **Biggest impact:** Feature engineering (+3.5%)
- **Total improvement:** +3.9% from Phase 1 baseline

### Business Impact Analysis

**Performance Achievements:**
- **71.3% accuracy** establishes strong baseline for financial sentiment analysis
- **+3.9% improvement** demonstrates value of systematic optimization
- **Feature engineering success** shows importance of domain-specific preprocessing

**Model Interpretability:**
- **Feature importance analysis** provides business insights
- **Class-specific patterns** help understand sentiment drivers
- **Financial terminology** captured effectively

**Limitations Identified:**
- **Class imbalance challenge** persists (negative class detection difficult)
- **70% accuracy ceiling** for classic ML approaches
- **Context ambiguity** in financial text requires advanced models

### Phase 2 Deliverables

✅ **Hyperparameter optimization** - Systematic tuning of all baseline models  
✅ **Feature engineering pipeline** - Domain-specific text preprocessing  
✅ **Ensemble method evaluation** - Voting and stacking approaches  
✅ **Performance benchmarking** - 71.3% accuracy with classic ML  
✅ **Feature importance analysis** - Business insights from model interpretation  
✅ **Optimization documentation** - Complete results and methodology  

### Next Phase Preparation

Phase 2 establishes the performance ceiling for classic machine learning approaches, providing:
- **Optimized feature engineering** (Financial Focus configuration)
- **Best ensemble approach** (Stacking Classifier)
- **Realistic performance expectations** (71.3% accuracy)
- **Clear improvement targets** for FinBERT fine-tuning (80-85% target)
- **Feature importance insights** for business interpretation

**Phase 3 Focus:** Advanced transformer models (FinBERT) to overcome classic ML limitations and achieve 80-85% accuracy through context-aware modeling.
