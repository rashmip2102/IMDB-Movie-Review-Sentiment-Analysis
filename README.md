# 🎬 IMDB Sentiment Analysis — NLP Pipeline

A step-by-step Natural Language Processing project on the IMDB 50K Movie Reviews dataset — covering text preprocessing, feature engineering, and word cloud visualization in preparation for sentiment classification.

---

## 📁 Project Structure

```
├── IMDB Dataset.csv                  # Raw dataset (50,000 movie reviews)
├── Preprocessed IMDB Dataset.csv     # Output of Step 1
├── Feature Engineered IMDB Dataset.csv  # Output of Step 2
├── PreprocessingIMDB.ipynb           # Step 1: Text preprocessing
├── FeatureEngineeringIMDB.ipynb      # Step 2: Feature engineering
└── Word_Cloud_IMDB.ipynb             # Step 3: Word cloud visualization
```

---

## 📊 Dataset

**Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle

| Column | Description |
|---|---|
| review | Raw movie review text |
| sentiment | Sentiment label (`positive` / `negative`) |

The dataset contains **50,000 reviews** evenly split between positive and negative sentiments, making it a balanced binary classification dataset.

---

## 📓 Notebooks

### 1. `PreprocessingIMDB.ipynb` — Text Preprocessing

Cleans raw review text through a full NLP preprocessing pipeline and exports `Preprocessed IMDB Dataset.csv`.

**Steps covered:**

- **HTML tag removal** — Uses `BeautifulSoup` with Python's built-in `html.parser` to strip all HTML tags (including `<script>` and `<style>` blocks) from raw reviews
- **Lowercasing** — Converts all text to lowercase for uniformity
- **Special character removal** — Uses regex (`re.sub`) to strip anything that isn't alphanumeric or a space
- **Tokenization** — Splits cleaned text into individual word tokens using NLTK's `word_tokenize`
- **Stemming** — Reduces tokens to their root forms using NLTK's `SnowballStemmer` (English)
- **Stop word removal** — Filters out common English stop words using spaCy's vocabulary (`nlp.vocab[token].is_stop`)
- Export cleaned dataset as `Preprocessed IMDB Dataset.csv`

---

### 2. `FeatureEngineeringIMDB.ipynb` — Feature Engineering

Extracts a rich set of numeric and text features from the preprocessed reviews and exports `Feature Engineered IMDB Dataset.csv`.

**Features extracted:**

**Bag of Words & N-grams** (via `sklearn.feature_extraction.text`)
- **Bag of Words** — `CountVectorizer` with top 15 features; prints vocabulary
- **Bi-gram** — `CountVectorizer` with `ngram_range=(2,2)`, top 15 features
- **Tri-gram** — `CountVectorizer` with `ngram_range=(3,3)`, top 15 features

**TF-IDF**
- `TfidfVectorizer` with top 100 features; prints IDF scores and feature names

**Linguistic Features** (per review, added as DataFrame columns)
- `feature word count` — Total number of words in the raw review
- `sentence length` — Total character count of the review
- `feature punctuation frequency` — Count of uppercase characters (proxy for emphasis/shouting)
- `feature readability` — Flesch Reading Ease score via `textstat`

**Lexical Features**
- `most_frequent_terms` — Top 3 most common tokens per review (from filtered review)
- `feature_real_word_ratio` — Ratio of tokens that appear in NLTK's English word corpus (real vs. nonsense words)

- Export enriched dataset as `Feature Engineered IMDB Dataset.csv`

---

### 3. `Word_Cloud_IMDB.ipynb` — Word Cloud Visualization

Generates a word cloud from the filtered (preprocessed) reviews to visually identify the most prominent terms across the entire corpus.

**Steps covered:**
- Loads `Feature Engineered IMDB Dataset.csv`
- Converts the `filtered review` column (stored as string-represented lists) back to plain text using `ast.literal_eval`
- Joins all reviews into a single text string
- Generates a word cloud (`800×400`, white background) using the `wordcloud` library
- Displays the result with `matplotlib`

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib scikit-learn nltk spacy beautifulsoup4 wordcloud textstat
```

Download the required NLTK and spaCy resources:

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
git clone https://github.com/rashmip2102/IMDB-Dataset.git
cd imdb-sentiment-analysis
jupyter notebook
```

Run the notebooks in this sequence:

1. `PreprocessingIMDB.ipynb` — requires `IMDB Dataset.csv`
2. `FeatureEngineeringIMDB.ipynb` — requires `Preprocessed IMDB Dataset.csv`
3. `Word_Cloud_IMDB.ipynb` — requires `Feature Engineered IMDB Dataset.csv`

> **Note:** Each notebook depends on the CSV exported by the previous one. Download the raw dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project root as `IMDB Dataset.csv` before running.

---

## 🔍 Key Findings

- **HTML noise** was significant in the raw reviews — `BeautifulSoup` was essential before any text analysis
- **Stemming + stop word removal** substantially reduced vocabulary size while retaining semantic signal
- **TF-IDF** outperforms raw Bag of Words for capturing term importance relative to the corpus
- **Word cloud** reveals that words like *film*, *movie*, *good*, *great*, *time*, and *story* dominate the corpus, reflecting typical review language
- **Real word ratio** provides a useful quality signal — low-ratio reviews may contain garbled HTML remnants or noise
