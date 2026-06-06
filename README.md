# 🎬 IMDB Sentiment Analysis — End-to-End NLP Pipeline

A complete Natural Language Processing project on the IMDB 50K Movie Reviews dataset — covering text preprocessing, feature engineering, visualization, and sentiment classification using five machine learning models.

---

## 📁 Project Structure

```
├── IMDB Dataset.csv                       # Raw dataset — 50,000 movie reviews (from Kaggle)
├── Preprocessed IMDB Dataset.csv          # Output of Step 1
├── Feature Engineered IMDB Dataset.csv    # Output of Step 2
├── Feature Engineered IMDB Dataset - 2.csv  # Output of Step 4 (post-modelling)
│
├── PreprocessingIMDB.ipynb                # Step 1: Text preprocessing
├── FeatureEngineeringIMDB.ipynb           # Step 2: Feature engineering
├── Word_Cloud_IMDB.ipynb                  # Step 3: Word cloud visualization
├── VisualizingTopFeaturesIMDB.ipynb       # Step 4: Feature importance & representation comparison
└── ML_Techmiques_Comparison_IMDB.ipynb   # Step 5: ML model training & comparison
```

---

## 📊 Dataset

**Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle

| Column | Description |
|---|---|
| `review` | Raw movie review text |
| `sentiment` | Sentiment label (`positive` / `negative`) |

The dataset contains **50,000 reviews** evenly split between positive and negative sentiments — a balanced binary classification benchmark.

---

## 📓 Notebooks

### 1. `Preprocessing.ipynb` — Text Preprocessing

Cleans the raw review text through a full NLP preprocessing pipeline and exports `Preprocessed IMDB Dataset.csv`.

**Steps covered:**
- **HTML tag removal** — `BeautifulSoup` with Python's `html.parser` strips all HTML tags, including `<script>` and `<style>` blocks
- **Lowercasing** — Converts all text to lowercase for uniformity
- **Special character removal** — `re.sub` strips anything that isn't alphanumeric or a space
- **Tokenization** — Splits cleaned text into individual word tokens via NLTK's `word_tokenize`
- **Stemming** — Reduces tokens to root forms using NLTK's `SnowballStemmer` (English)
- **Stop word removal** — Filters common English stop words using spaCy's vocabulary (`nlp.vocab[token].is_stop`)
- Exports `Preprocessed IMDB Dataset.csv`

---

### 2. `Feature Engineering.ipynb` — Feature Engineering

Extracts a rich set of text and numeric features from the preprocessed reviews and exports `Feature Engineered IMDB Dataset.csv`.

**Text representation features** (via `sklearn`):
- **Bag of Words** — `CountVectorizer`, top 15 features
- **Bi-gram** — `CountVectorizer` with `ngram_range=(2,2)`, top 15 features
- **Tri-gram** — `CountVectorizer` with `ngram_range=(3,3)`, top 15 features
- **TF-IDF** — `TfidfVectorizer`, top 100 features with IDF scores

**Linguistic features** (added as DataFrame columns per review):
| Column | Description |
|---|---|
| `feature word count` | Total word count of the raw review |
| `sentence length` | Total character count of the review |
| `feature punctuation frequency` | Count of uppercase characters (proxy for emphasis) |
| `feature readability` | Flesch Reading Ease score via `textstat` |

**Lexical features:**
| Column | Description |
|---|---|
| `most_frequent_terms` | Top 3 most common tokens per review |
| `feature_real_word_ratio` | Ratio of tokens found in NLTK's English word corpus |

- Exports `Feature Engineered IMDB Dataset.csv`

---

### 3. `Word Cloud.ipynb` — Word Cloud Visualization

Generates a word cloud across the entire corpus to identify the most prominent terms visually.

**Steps covered:**
- Loads `Feature Engineered IMDB Dataset.csv`
- Converts `filtered review` (stored as string-represented lists) back to plain text using `ast.literal_eval`
- Joins all reviews into a single corpus string
- Generates an `800×400` word cloud (white background) using the `wordcloud` library
- Renders with `matplotlib`

---

### 4. `Visualizing Top Features.ipynb` — Feature Importance & Representation Comparison

Two visualizations that reveal what the model learns and which text representation strategy works best.

**Top sentiment-driving words:**
- Fits a `TfidfVectorizer` (5,000 features) and a `LogisticRegression` model on `filtered review`
- Extracts model coefficients and identifies the **top 10 positive** and **top 10 negative** words by weight
- Plots a horizontal bar chart with green bars for positive words and red bars for negative words

**Feature representation comparison:**
- Benchmarks four text representations — **BoW**, **Bi-gram**, **Tri-gram**, and **TF-IDF** — each with 5,000 features
- Trains a `LogisticRegression` classifier (80/20 split, stratified) for each representation
- Compares **Accuracy** and **F1-Score** side-by-side in a grouped bar chart

---

### 5. `ML Techniques Comparison.ipynb` — ML Model Training & Comparison

Trains and benchmarks five classifiers on a combined feature set (TF-IDF + engineered numeric features).

**Feature matrix construction:**
- **Text features** — `TfidfVectorizer` with 5,000 features on `cleaned_review`
- **Numeric features** — `feature word count`, `sentence length`, `feature punctuation frequency`, `feature readability`, `feature_real_word_ratio` scaled with `MinMaxScaler`
- Both feature sets combined using `scipy.sparse.hstack` to keep the matrix sparse

**Train / Validation / Test split:**
- 70% training / 15% validation / 15% test (stratified on sentiment)

**Models trained:**
| Model | Library |
|---|---|
| Logistic Regression | `sklearn` — `max_iter=1000` |
| Naive Bayes | `sklearn` — `MultinomialNB` |
| Support Vector Machine | `sklearn` — `LinearSVC` |
| Random Forest | `sklearn` — `RandomForestClassifier` |
| XGBoost | `xgboost` — `XGBClassifier` |

**Evaluation metrics** (on validation set): Accuracy, Precision, Recall, F1-Score, and Confusion Matrix per model.

- Exports `Feature Engineered IMDB Dataset - 2.csv`

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy beautifulsoup4 wordcloud textstat xgboost
```

Download the required NLTK data and spaCy model:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
```

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/rashmip2102/IMDB-Sentiment-Analysis-End-to-End-NLP-Pipeline.git
cd imdb-sentiment-analysis
jupyter notebook
```

Run the notebooks in this order:

1. `PreprocessingIMDB.ipynb` — requires `IMDB Dataset.csv`
2. `FeatureEngineeringIMDB.ipynb` — requires `Preprocessed IMDB Dataset.csv`
3. `Word_Cloud_IMDB.ipynb` — requires `Feature Engineered IMDB Dataset.csv`
4. `VisualizingTopFeaturesIMDB.ipynb` — requires `Feature Engineered IMDB Dataset.csv`
5. `ML_Techmiques_Comparison_IMDB.ipynb` — requires `Feature Engineered IMDB Dataset.csv`

> **Note:** Download the raw dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project root as `IMDB Dataset.csv` before running. Notebooks 3, 4, and 5 can be run in any order after notebook 2.

---

## 🔍 Key Findings

- **HTML noise** was pervasive in raw reviews — `BeautifulSoup` was essential before any analysis
- **TF-IDF** consistently outperforms raw Bag of Words and N-grams for sentiment classification
- **Top positive words** driving sentiment include terms like *great*, *excellent*, *wonderful*; **top negative words** include *worst*, *awful*, *waste*
- **Combining TF-IDF with engineered numeric features** (readability, word count, real word ratio) improves classifier performance over text alone
- **SVM and Logistic Regression** are the strongest performers among the five models evaluated; **Naive Bayes** trains fastest but with a lower accuracy ceiling
- **Real word ratio** serves as a useful quality signal — reviews with low ratios often contain residual HTML or garbled text
