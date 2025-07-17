# 📰 Fake and Real News Detection Using Deep Learning

This project focuses on automatically detecting **fake** versus **real** news articles written in **English** using Natural Language Processing (NLP) and Deep Learning techniques.  
The dataset comes from the well-known **Fake and Real News** dataset available on Kaggle. The goal is to build a robust deep learning model to classify news articles accurately and help combat misinformation.

---

## 🎯 Motivation

The proliferation of fake news has a significant negative impact on public opinion and decision-making worldwide.  
This project aims to:
- Analyze the linguistic differences between fake and real news articles  
- Apply state-of-the-art deep learning techniques for text classification  
- Gain hands-on experience in text preprocessing, model building, and evaluation  

---

## 🗃️ Dataset

The dataset used in this project is the **Fake and Real News Dataset** available on [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).  
It contains labeled news articles split into two classes:

- **Label 0**: Fake News  
- **Label 1**: Real News  

Columns include: `title`, `text`, and the corresponding label.

---

## ⚙️ Project Workflow

1. **Data Cleaning**  
   - Removal of HTML tags, digits, punctuation, emails, URLs, and stopwords  
   - Lemmatization using `nltk`  
   - Filtering out very short and very long news articles to maintain quality  

2. **Exploratory Data Analysis (EDA)**  
   - Statistical analysis of text lengths and distributions  
   - Visualization with `seaborn` and `matplotlib` to understand dataset characteristics  

3. **Text Vectorization**  
   - Tokenization with Keras `Tokenizer`  
   - Padding sequences to a fixed maximum length (200 tokens)  

4. **Model Architecture**  
   - Embedding layer to learn word representations  
   - Bidirectional LSTM to capture contextual dependencies in both directions  
   - Dropout layers to prevent overfitting  
   - Dense output layer with sigmoid activation for binary classification  

5. **Training & Evaluation**  
   - Split dataset into training and testing (80/20)  
   - Use binary cross-entropy loss and accuracy metric  
   - Train for 5 epochs with batch size 64  

---

## 🧠 Model Summary

Embedding → Bidirectional LSTM → Dropout → Dense (ReLU) → Dropout → Dense (Sigmoid)


- Optimizer: Adam with learning rate 0.0001  
- Epochs: 5  
- Batch size: 64  

---

## 📈 Results

| Metric          | Value  |
|-----------------|---------|
| Training Accuracy | ~XX%    |
| Testing Accuracy  | ~YY%    |
| Loss             | ~ZZ     |

_(Replace placeholders with actual results)_

---

## 💡 Future Improvements

- Experiment with pre-trained transformer-based models such as **BERT**, **RoBERTa**, or domain-specific models  
- Hyperparameter tuning for better performance  
- Incorporate news metadata like titles and publishing sources  
- Data augmentation techniques to increase dataset diversity  

---

## 📚 Libraries Used

- Python 3.10  
- TensorFlow / Keras  
- NLTK  
- Seaborn  
- Pandas  
- Matplotlib  

---

## 📁 Project Structure

FakeNewsDetection/
├── data/ # Dataset CSV files
├── notebooks/ # Jupyter notebooks with experiments
├── models/ # Saved trained model files
├── scripts/ # Utility and preprocessing scripts
├── README.md
└── requirements.txt


---

## 👨‍💻 Author

**Mohammad Jafa (JafaDsX)**  
Self-taught Data Scientist with passion for NLP and deep learning 🐍📊  
[GitHub](https://github.com/JafaDsX) | [Kaggle](https://www.kaggle.com/jafadsx) | [Email](mailto:jafadsx@gmail.com)

---

# FakeNewsDetection
# FakeNewsDetection
