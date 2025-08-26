# Financial Sentiment Analysis for Investment Signal Generation

## 1. Executive Summary

### Business Problem
In quantitative finance, the velocity and volume of news make manual analysis impossible. Investment firms need automated solutions to gauge market sentiment from financial headlines to inform trading strategies. Delays or inaccuracies in interpreting news can lead to missed opportunities or significant losses. This project tackles this challenge by building and comparing machine learning models to classify financial news sentiment, with the ultimate goal of creating a high-fidelity signal for algorithmic trading.

### Solution & Key Findings
This project systematically progressed through three phases of model development, starting with simple baselines and culminating in a state-of-the-art transformer model:
- **Phase 1 (Baselines):** Established initial benchmarks using TF-IDF and Logistic Regression, achieving **67% accuracy**.
- **Phase 2 (Classic ML):** Improved upon baselines with advanced feature engineering and ensemble methods (Stacking), reaching **71% accuracy**.
- **Phase 3 (FinBERT):** Deployed a domain-specific transformer, `ProsusAI/finbert`, which was fine-tuned to achieve **81% accuracy**, significantly outperforming all other models.

The key finding is that while traditional ML models provide a reasonable signal, **only a domain-specific transformer like FinBERT can achieve the performance required for a reliable production system**, especially in accurately detecting negative sentiment—a crucial factor for risk management.

---

## 2. Key Achievements & Business KPIs

The project successfully delivered a model that provides a significant analytical edge. The progression from Phase 1 to Phase 3 yielded transformative improvements:

| Metric                  | Baseline (Phase 1) | Final Model (Phase 3) | Improvement                             | Business Value                               |
| ----------------------- | ------------------ | --------------------- | --------------------------------------- | -------------------------------------------- |
| **Overall Accuracy**    | 67.4%              | **80.8%**             | **+13.4 percentage points**             | More reliable signals for trading algorithms. |
| **Negative F1 Score**   | 7%                 | **50%**               | **+43 percentage points (~7x)**         | Superior detection of negative, risk-bearing news. |
| **Model Complexity**    | Low                | High                  | -                                       | Justified by the massive performance leap.      |

![Business KPIs](/.github/assets/kpis.png)

---

## 3. Solution Overview: A Three-Phase Approach

The project was structured in three distinct phases to ensure a rigorous, iterative development process.

### Phase 1: Baseline Models
- **Objective:** Establish a performance baseline using standard NLP techniques.
- **Method:** TF-IDF vectorization paired with several classic classifiers (Logistic Regression, Naive Bayes, SVM).
- **Outcome:** The best model (Logistic Regression) achieved **67.4% accuracy**. This phase highlighted the difficulty of the task, especially the model's inability to correctly identify negative headlines (7% F1-score).

### Phase 2: Optimized Classical ML
- **Objective:** Maximize the performance of traditional models through feature engineering and ensembling.
- **Method:**
    - **Feature Engineering:** Optimized TF-IDF by removing stop words and tuning n-grams, tailored to financial vocabulary.
    - **Ensemble Methods:** Implemented a Stacking Classifier that combined Logistic Regression, SVM, and Random Forest.
- **Outcome:** Performance improved to **71.3% accuracy**. While an improvement, the model still struggled with negative sentiment, indicating the limits of this approach.

### Phase 3: Fine-Tuning FinBERT
- **Objective:** Leverage a state-of-the-art, domain-specific transformer model to break through the performance ceiling.
- **Method:** Fine-tuned `ProsusAI/finbert`, a BERT model pre-trained on a massive corpus of financial text.
- **Outcome:** A breakthrough in performance, achieving **80.8% accuracy**. Most importantly, the F1-score for the negative class jumped from 7% to **50%**, a **7x improvement** that makes the model truly valuable for risk assessment.

---

## 4. Performance Analysis & ROI

The central story of this project is the trade-off between complexity and performance. While FinBERT is more computationally expensive, its superior performance provides a clear return on investment.

### Model Performance Dashboard
The dashboard below visualizes the journey from the simple baseline to the high-performance FinBERT model. The most significant jump in both accuracy and risk detection capability occurred in Phase 3.

![Performance Dashboard](/.github/assets/dashboard.png)

### Justifying the Investment in Transformers
The analysis shows that the move from classic ML to a transformer model delivered the biggest ROI. The final model is not just incrementally better; it is qualitatively different, providing a level of reliability that the simpler models could not approach. For an investment firm, this reliability is the difference between a toy model and a production-ready trading signal.

![Performance vs Complexity](/.github/assets/roi.png)

---

## 5. How to Run This Project

### Prerequisites
- Python 3.9+
- pip for package management

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/jp-marques/financial-sentiment-analysis.git
    cd financial-sentiment-analysis
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebooks
The project is organized into notebooks that correspond to the project phases. They should be run in the following order:
1.  `notebooks/01_eda_baselines.ipynb`
2.  `notebooks/02_classic_ml.ipynb`
3.  `notebooks/03_finetune_finbert.ipynb`
4.  `notebooks/04_visualizations.ipynb`

---

## 6. Data Source

The dataset used in this project is the **Financial Sentiment Analysis Dataset**, which is widely used for financial sentiment analysis tasks. It contains sentences from financial news categorized by sentiment (Positive, Negative, Neutral).

- **Source:** [Financial Sentiment Analysis on Kaggle](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)
- **License:** The dataset is available under the [Creative Commons CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

The data was split into training, validation, and test sets as detailed in the EDA notebook (`notebooks/01_eda_baselines.ipynb`). No additional filtering was applied beyond the standard preprocessing steps outlined in the notebooks.

---

## 7. Tech Stack
- **Data Analysis & Modeling:** Pandas, Scikit-learn, PyTorch
- **NLP:** Hugging Face Transformers (`ProsusAI/finbert`), NLTK
- **Visualizations:** Matplotlib, Seaborn, Plotly
- **Environment:** Python, Jupyter Notebooks

---

## 8. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
