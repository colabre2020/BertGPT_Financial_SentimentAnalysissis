# 🔍 Unlocking the Market: Harnessing Large Language Models for Predicting Stock Movements through Sentiment Analysis

This project explores how **Large Language Models (LLMs)** like **BERT** and **GPT-2** can be used for sentiment analysis on financial news headlines, and how this sentiment can be correlated to predict stock price movements.

---

## 📘 Abstract

This research investigates the application of BERT and GPT-based models in financial sentiment classification. It compares their performance with traditional machine learning models such as Logistic Regression and SVM, using a labeled dataset of financial news headlines. The goal is to evaluate the effectiveness of modern LLMs in extracting market-relevant sentiment and discuss their potential utility in predictive stock analysis.

---

## 📂 Dataset

- **Source:** [Kaggle - Sentiment for Financial News Dataset](https://www.kaggle.com/datasets/waseemalastal/sentiment-for-financial-news-dataset)
- **Content:**
  - Financial news headlines
  - Labeled sentiment: Positive, Neutral, Negative
  - Associated stock tickers (no price data included)

---

## 🧠 Models Used

### 1. BERT (Bidirectional Encoder Representations from Transformers)
- Model: `bert-base-uncased`
- Fine-tuned for sentiment classification
- Strong bidirectional context understanding

### 2. GPT-2 (Generative Pretrained Transformer)
- Model: `gpt2`
- Used in zero-shot and fine-tuned configurations
- Evaluated for classification accuracy despite generative design

### 3. Traditional Baselines
- TF-IDF + Logistic Regression
- TF-IDF + Support Vector Machine (SVM)

---

## ⚙️ Preprocessing

- Null value removal
- Lowercasing, punctuation stripping
- BERT & GPT-specific tokenization
- Label encoding

---

## 📊 Results

| Model                | Accuracy | F1-Score |
|---------------------|----------|----------|
| BERT                | 86%      | 0.86     |
| GPT-2               | 84%      | 0.84     |
| Logistic Regression | 83%      | 0.80     |
| SVM                 | 85%      | 0.82     |

BERT outperforms the other models due to its robust context understanding. GPT-2 shows strong generalization, while traditional models offer competitive baselines.

---

## 📈 Visualizations

- Sentiment class distributions
- Confusion matrices for all models
- Model comparison bar plots

Plots are located in the `images/` directory.

---

## 📁 Repository Structure

