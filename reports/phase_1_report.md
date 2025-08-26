# Financial Sentiment Analysis Project Report

## Phase 1: Data Exploration & Preprocessing

### Dataset Overview
We utilized the **Kaggle Financial Sentiment Analysis** dataset containing 5,842 financial headlines and sentences. The dataset provides a clean, pre-labeled foundation for sentiment analysis with three distinct classes: positive, neutral, and negative.

**Key Characteristics:**
- **Total samples:** 5,842 financial headlines
- **Data quality:** No missing values or encoding issues
- **Text format:** Financial news headlines, earnings reports, and market commentary
- **Label distribution:** 
  - Neutral: 53.6% (3,131 samples)
  - Positive: 31.7% (1,852 samples) 
  - Negative: 14.7% (859 samples)

### Data Structure Analysis
The dataset contains two primary columns:
- **Sentence:** Raw financial text (headlines, statements, market updates)
- **Sentiment:** Three-class sentiment labels (positive, neutral, negative)

**Text Length Statistics:**
- **Average length:** 117 characters
- **Range:** 9 to 315 characters
- **Distribution:** Most headlines fall within 50-150 character range, indicating consistent formatting typical of financial news sources.

### Data Quality Assessment
- **Completeness:** 100% - no missing values in either text or label columns
- **Consistency:** Labels are already standardized to the target three-class format
- **Balance:** Dataset shows natural class imbalance reflecting real-world financial sentiment distribution
- **Representativeness:** Covers diverse financial topics including earnings, market movements, and corporate announcements

### Train/Validation/Test Splits
We implemented a **stratified 70/15/15 split** to maintain class distribution across all subsets:

| Split | Size | Neutral | Positive | Negative |
|-------|------|---------|----------|----------|
| **Train** | 4,089 (70%) | 53.6% | 31.7% | 14.7% |
| **Validation** | 876 (15%) | 53.5% | 31.7% | 14.7% |
| **Test** | 877 (15%) | 53.6% | 31.7% | 14.7% |

**Key Benefits:**
- Stratified sampling ensures representative class distribution
- Sufficient validation set for hyperparameter tuning
- Adequate test set for final model evaluation
- Random state (42) ensures reproducibility

### Representative Examples
**Positive:** "Stora Enso owns 43 percent of Bergvik and earns therefore SEK 1.5 bn on the value appreciation..."

**Neutral:** "The company serves customers in various industries, including process and resources, industrial manufacturing..."

**Negative:** "$AAPL weekly still under the 50 moving average and creating a lower high..."

### Business Context
This dataset provides a realistic foundation for financial sentiment analysis, capturing the natural distribution of market sentiment where neutral reporting dominates, positive news is more common than negative, and text lengths are optimized for financial communication channels.

### Phase 1 Deliverables
✅ **Data pipeline established** - Clean dataset loaded and validated  
✅ **Label mapping confirmed** - Three-class sentiment structure ready  
✅ **Data quality verified** - No missing values or encoding issues  
✅ **Stratified splits created** - Train/validation/test sets with preserved class balance  
✅ **Basic EDA completed** - Class distribution and text characteristics documented  

### Baseline Model Performance
We implemented seven baseline models using TF-IDF features to establish performance benchmarks:

**Model Performance (Validation Set):**
- **Naive Bayes:** 69.7% accuracy 🏆
- **Logistic Regression:** 68.3% accuracy
- **SVM:** 68.2% accuracy
- **Gradient Boosting:** 67.4% accuracy
- **Linear SVM:** 65.5% accuracy
- **Random Forest:** 65.3% accuracy
- **Decision Tree:** 56.2% accuracy

**Key Insights:**
- **Class imbalance challenge:** Negative class recall only 4% (83/129) despite 83% precision
- **Neutral dominance:** 98% recall for neutral class, reflecting dataset distribution
- **Feature effectiveness:** TF-IDF captures financial terminology well for neutral/positive classes
- **Baseline ceiling:** ~70% accuracy establishes realistic improvement targets for FinBERT

### Next Phase Preparation
Phase 1 establishes the foundation for advanced model development, providing:
- Clean, preprocessed text data
- Balanced train/validation/test splits
- Understanding of class imbalance challenges
- **Baseline performance benchmarks (69.7% best accuracy)**
- Clear improvement targets for FinBERT fine-tuning