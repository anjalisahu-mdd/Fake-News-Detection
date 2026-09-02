# Fake News Detection Using NLP & Machine Learning

## Overview

This project implements a **Fake News Detection** pipeline using **Natural Language Processing (NLP)** and a **Multinomial Naive Bayes** classifier. The model processes news titles, converts the cleaned text into numerical features, and classifies the news into **FAKE** or **REAL** categories.

## Project Workflow

```text
News Dataset (train.csv)
        ↓
Data Cleaning
        ↓
Remove Missing Values
        ↓
Text Preprocessing
        ├── Remove non-alphabetic characters
        ├── Convert to lowercase
        ├── Tokenization
        ├── Stop-word removal
        └── Porter Stemming
        ↓
CountVectorizer
(1–3 gram features, max 5000 features)
        ↓
Train/Test Split
(67% training / 33% testing)
        ↓
Multinomial Naive Bayes
        ↓
FAKE / REAL Prediction
        ↓
Accuracy + Confusion Matrix
```

## Dataset

The project expects a file named:

```text
train.csv
```

The code uses the **`title`** field as the text input and **`label`** as the target variable.

Expected target classes:

- `FAKE`
- `REAL`

Missing rows are removed before text preprocessing.

## Text Preprocessing

The news titles are processed using **NLTK** and Python regular expressions:

1. Remove non-alphabetic characters.
2. Convert text to lowercase.
3. Split text into individual words.
4. Remove English stopwords.
5. Apply **Porter Stemming**.
6. Recombine the processed tokens into cleaned text.

## Feature Extraction

The project uses **Scikit-learn CountVectorizer**:

- Maximum features: **5000**
- N-gram range: **1–3**
- Features include unigrams, bigrams, and trigrams.
- The processed text is transformed into a numerical feature matrix.

## Machine Learning Model

A **Multinomial Naive Bayes** classifier is trained on the extracted text features.

The dataset is divided using:

- **67% training data**
- **33% testing data**
- `random_state = 0`

## Evaluation

Model performance is evaluated using:

- **Accuracy Score**
- **Confusion Matrix**

The confusion matrix compares actual and predicted `FAKE` / `REAL` labels.

> Note: The project code calculates the accuracy, but this README intentionally does not claim a numerical accuracy because the verified output value is not provided in the source file.

## Technologies Used

- **Python**
- **Pandas**
- **NLTK**
- **Scikit-learn**
- **NumPy**
- **Matplotlib**
- **Regular Expressions (Regex)**

## Key Concepts

- Natural Language Processing
- Text preprocessing
- Tokenization
- Stop-word removal
- Porter stemming
- N-gram feature extraction
- Count-based text vectorization
- Multinomial Naive Bayes
- Binary text classification
- Confusion matrix evaluation

## Project Structure

```text
Fake-News-Detection/
│
├── train.csv
├── Fake_News_Detection.py
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Fake-News-Detection
```

### 2. Install dependencies

```bash
pip install pandas numpy nltk scikit-learn matplotlib
```

### 3. Add the dataset

Place `train.csv` in the project directory.

### 4. Run the project

Run the Python script/notebook containing the Fake News Detection pipeline.

The program will preprocess the news titles, train the classifier, generate predictions, calculate accuracy, and display the confusion matrix.

## Important Implementation Details

The core feature-extraction configuration is:

```python
CountVectorizer(max_features=5000, ngram_range=(1,3))
```

The classifier is:

```python
MultinomialNB()
```

The model is evaluated with:

```python
metrics.accuracy_score(y_test, pred)
```

and:

```python
metrics.confusion_matrix(y_test, pred)
```

## Future Improvements

Possible extensions include:

- Compare **TF-IDF** with CountVectorizer.
- Compare Naive Bayes with other classification algorithms.
- Add precision, recall, and F1-score.
- Build a user interface for entering a news headline.
- Add model persistence for reuse without retraining.

## Disclaimer

This repository describes the implementation contained in the provided source code. Any future modification, retraining, dataset change, or model improvement should be documented separately.
