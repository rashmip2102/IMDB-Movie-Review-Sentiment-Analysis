# 🎬 IMDB Movie Review Sentiment Analysis

A complete end-to-end NLP pipeline for binary sentiment classification on the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The project covers everything from raw text preprocessing to hyperparameter-tuned ML models, with rich visualizations at each stage.

---

## 📁 Project Structure

```
├── Preprocessing.ipynb               # Text cleaning, tokenization, stemming, stop word removal
├── Feature Engineering.ipynb          # BoW, N-grams, TF-IDF, linguistic features
├── Word Cloud.ipynb                 # Word cloud visualization of processed reviews
├── Visualizing Top Features.ipynb      # Top sentiment-driving words & feature comparison charts
├── ML Techniques Comparison.ipynb   # Training & comparing 5 ML classifiers
├── Optimized ML Pipeline and Tuning.ipynb # Hyperparameter tuning & final optimized pipeline
```

---

## 📊 Dataset

- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews
- **Labels:** Binary — `positive` / `negative` (balanced, 25K each)
- **Columns:** `review`, `sentiment`

---

## 🔧 Pipeline Overview

### 1. Preprocessing
Raw reviews go through a multi-step cleaning pipeline:
- HTML tag removal using `BeautifulSoup`
- Lowercasing and special character removal via regex
- Tokenization with `nltk`
- Stemming using `SnowballStemmer`
- Stop word removal using `spaCy`

Output saved as `Preprocessed IMDB Dataset.csv`.

### 2. Feature Engineering
Multiple text representation strategies are applied:
- **Bag of Words (BoW)** — unigram term frequency
- **N-grams** — bigram and trigram representations
- **TF-IDF** — term frequency–inverse document frequency (top 100 features)
- **Linguistic Features** — word count, sentence length, punctuation frequency, capitalisation count, Flesch readability score, real-word ratio

Output saved as `Feature Engineered IMDB Dataset.csv`.

### 3. Word Cloud
Generates a word cloud from the preprocessed and filtered reviews to visualise the most frequent terms in the corpus.

### 4. Visualizing Top Features
- Bar chart of the **top 10 positive and top 10 negative words** driving sentiment, based on Logistic Regression coefficients
- **Feature representation comparison** — accuracy and F1-score of BoW, Bigrams, Trigrams, and TF-IDF across a Logistic Regression baseline

### 5. ML Techniques Comparison
Five classifiers are trained on a **combined feature matrix** (TF-IDF + scaled linguistic features) with a 70/15/15 train-validation-test split:

| Model | Notes |
|---|---|
| Logistic Regression | Regularised, `max_iter=1000` |
| Naive Bayes | Multinomial; sparse-compatible |
| SVM | `LinearSVC` with auto dual |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting classifier |

Models are evaluated on Accuracy, Precision, Recall, and F1-Score on the validation set, with confusion matrices printed for each.

### 6. Optimized ML Pipeline & Tuning 
Hyperparameter tuning and final pipeline optimization on the best-performing models from the comparison stage.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
```

### Key Dependencies

```
pandas
numpy
scikit-learn
xgboost
nltk
spacy
beautifulsoup4
textstat
wordcloud
matplotlib
seaborn
scipy
```

Download the required NLTK data and spaCy model before running:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# spaCy English model
# python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

Run the notebooks in order:

1. `Preprocessing.ipynb`
2. `Feature Engineering.ipynb`
3. `Word Cloud.ipynb` *(optional, visualization)*
4. `Visualizing Top Features.ipynb` *(optional, visualization)*
5. `ML Techniques Comparison.ipynb`
6. `Optimized ML Pipeline and Tuning.ipynb`

> Make sure the dataset CSV is in the same directory as the notebooks, or update the file paths accordingly.

---

## 📈 Results

The ML comparison notebook evaluates all five models on the validation set across Accuracy, Precision, Recall, and F1-Score. The feature comparison notebook additionally benchmarks BoW, Bigram, Trigram, and TF-IDF representations, providing insight into which text encoding strategy best captures sentiment signal.

---

## 📄 License

This project is for educational purposes. The dataset is sourced from Kaggle and subject to its original license.
