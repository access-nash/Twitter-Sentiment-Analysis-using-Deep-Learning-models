import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import torch
import time
from collections import defaultdict
import seaborn as sns
from gensim.models import Word2Vec
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import (Embedding, LSTM, GRU, SimpleRNN,
                         Dense, Dropout, Bidirectional)
from keras.callbacks import EarlyStopping
import time
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from pathlib import Path

nltk.download('punkt_tab')


# Data Preparation
def load_and_preprocess_data():
    """Load and preprocess the Twitter sentiment dataset"""
    df = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/NLP with Pytorch/training.1600000.processed.noemoticon.csv', encoding = 'latin1')
    df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

    # Convert to binary classification (0=negative, 1=positive)
    df['polarity'] = df['polarity'].replace(4, 1)

    # Clean text
    def clean_text(text):
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase and remove extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    # Sample 50k points for faster training
    #df = df.sample(50000, random_state=42)

    return df

df = load_and_preprocess_data()
print("\nMissing Values:\n", df.isnull().sum())

# Split data into train (70%), validation (15%), test (15%)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['polarity'], test_size=0.3, random_state=42, stratify=df['polarity'])
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

print("\nLabel distribution verification:")
print("Full dataset:", np.unique(df['polarity'], return_counts=True))
print("Training set:", np.unique(y_train, return_counts=True))
print("Validation set:", np.unique(y_val, return_counts=True))
print("Test set:", np.unique(y_test, return_counts=True))

# BERT Tokenization
bert_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')


def bert_tokenize(texts, max_len=64):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)

# Tokenize all splits
X_train_bert, train_masks = bert_tokenize(X_train)
X_val_bert, val_masks = bert_tokenize(X_val)
X_test_bert, test_masks = bert_tokenize(X_test)

# Word Embeddings for RNN models
nltk.download('punkt')

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    pattern = r"[a-zA-Z\s]"
    text = ''.join(re.findall(pattern, text))
    text = text.lower()
    return nltk.word_tokenize(text)

train_tokens = X_train.apply(preprocess_text)
val_tokens = X_val.apply(preprocess_text)
test_tokens = X_test.apply(preprocess_text)

# Train Word2Vec embeddings
word2vec_model = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    min_count=1,
    workers=4,
    window=5,
    epochs=10
)

# Create embedding matrix
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv and i < 20000:
        embedding_matrix[i] = word2vec_model.wv[word]

# Convert texts to sequences and pad
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)


#  BERT Model
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
# import torch

from transformers import BertTokenizer, BertForSequenceClassification

class BertSentimentClassifier:
    def __init__(self, n_classes=2):
        self.model = BertForSequenceClassification.from_pretrained(
            'cardiffnlp/twitter-roberta-base-sentiment',
            num_labels=n_classes
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


    def train(self, train_loader, val_loader, epochs=4, lr=3e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct_preds = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs.logits, dim=1)
                correct_preds += torch.sum(preds == labels).item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_preds / len(data_loader.dataset)
        return avg_loss, accuracy

    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return true_labels, predictions


# RNN-based Models
def build_rnn_model(model_type, embedding_matrix, vocab_size, max_len):
    model = Sequential()

    model.add(Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False
    ))

    if model_type == 'LSTM':
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    elif model_type == 'GRU':
        model.add(Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2)))
    elif model_type == 'RNN':
        model.add(SimpleRNN(128, dropout=0.2, return_sequences=True))
        model.add(SimpleRNN(64, dropout=0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()]
    )

    return model


# BERT data loaders
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)
test_labels = torch.tensor(y_test.values)

train_data = TensorDataset(
    torch.tensor(X_train_bert),
    torch.tensor(train_masks),
    train_labels
)
val_data = TensorDataset(
    torch.tensor(X_val_bert),
    torch.tensor(val_masks),
    val_labels
)
test_data = TensorDataset(
    torch.tensor(X_test_bert),
    torch.tensor(test_masks),
    test_labels
)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

#  results dictionary
results = {
    'BERT': {},
    'LSTM': {},
    'GRU': {},
    'RNN': {}
}

# Train and evaluate BERT
print("\n*** Training BERT ***")
bert_model = BertSentimentClassifier()
bert_train_metrics = bert_model.train(
    train_loader,
    val_loader,
    epochs=4,
    lr=2e-5
)

# Evaluate BERT
true_labels, pred_labels = bert_model.predict(test_loader)

if true_labels is None or pred_labels is None:
    raise ValueError("BERT model returned None predictions")

results['BERT'] = {
    'accuracy': accuracy_score(true_labels, pred_labels),
    'report': classification_report(true_labels, pred_labels, target_names=['Negative', 'Positive']),
    'cm': confusion_matrix(true_labels, pred_labels),
    'roc_auc': roc_auc_score(true_labels, pred_labels),
    'params': sum(p.numel() for p in bert_model.model.parameters())
}

# Train and evaluate RNN models
for model_type in ['LSTM', 'GRU', 'RNN']:
    print(f"\n*** Training {model_type} ***")
    model = build_rnn_model(model_type, embedding_matrix, vocab_size, max_len)

    start_time = time.time()
    history = model.fit(
        X_train_pad, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_val_pad, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
        verbose=1
    )
    train_time = time.time() - start_time

    # Evaluate
    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

    results[model_type] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, target_names=['Negative', 'Positive']),
        'cm': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'history': history.history,
        'train_time': train_time,
        'params': model.count_params()
    }

    print(f"\n=== {model_type} Evaluation ===")
    print("Accuracy:", results[model_type]['accuracy'])
    print("Classification Report:")
    print(results[model_type]['report'])
    print("Confusion Matrix:")
    print(results[model_type]['cm'])
    print("ROC-AUC Score:", results[model_type]['roc_auc'])

# Visualization
import matplotlib

matplotlib.use('Agg') # some interactive forms in Pycharm causing plots to hang

# Bar chart comparing F1-scores for all models
plt.figure(figsize=(10, 5))
f1_scores = []
model_names = []

for model_name, model_results in results.items():
    report = model_results['report']

    if isinstance(report, str):
        for line in report.split('\n'):
            if 'weighted avg' in line:
                parts = line.split()
                f1_score = float(parts[-2])
                f1_scores.append(f1_score)
                model_names.append(model_name)
                break

sns.barplot(x=model_names, y=f1_scores)
plt.title('Model Comparison by F1-Score')
plt.ylabel('Weighted F1-Score')
plt.ylim(0, 1)
plt.savefig('f1_scores.png', bbox_inches='tight', dpi=300)
plt.close()

# Training curves

# BERT training curves

if 'BERT' in results and 'history' in results['BERT']:
    plt.figure(figsize=(8, 5))
    if 'loss' in results['BERT']['history']:
        plt.plot(results['BERT']['history']['loss'], label='Train Loss')
    if 'val_loss' in results['BERT']['history']:
        plt.plot(results['BERT']['history']['val_loss'], label='Val Loss')
    plt.title('BERT Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('bert_training.png', bbox_inches='tight', dpi=300)
    plt.close()

# RNN models training curves
for model_name in ['LSTM', 'GRU', 'RNN']:
    if model_name in results and 'history' in results[model_name]:
        plt.figure(figsize=(8, 5))
        history = results[model_name]['history']
        if 'loss' in history:
            plt.plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], '--', label='Val Loss')
        plt.title(f'{model_name} Training Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_name.lower()}_training.png', bbox_inches='tight', dpi=300)
        plt.close()

# Confusion matrices
for model_name in results:
    plt.figure(figsize=(6, 5))
    cm = results[model_name]['cm']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cmap='Blues'
    )
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name.lower()}_cm.png', bbox_inches='tight', dpi=300)
    plt.close()

# Consolidated Performance summary
summary_data = []

for model_name, model_results in results.items():

    precision = recall = f1_score = None

    report = model_results.get('report')
    if isinstance(report, str):
        for line in report.split('\n'):
            if 'weighted avg' in line:
                parts = [x for x in line.split() if x]
                try:
                    precision = float(parts[-4])
                    recall = float(parts[-3])
                    f1_score = float(parts[-2])
                except (IndexError, ValueError):
                    pass
                break

    train_time = model_results.get('train_time')

    summary_data.append({
        'Model': model_name,
        'Accuracy': model_results.get('accuracy', float('nan')),
        'Precision': precision if precision is not None else float('nan'),
        'Recall': recall if recall is not None else float('nan'),
        'F1-Score': f1_score if f1_score is not None else float('nan'),
        'ROC-AUC': model_results.get('roc_auc', float('nan')),
        'Training Time (s)': f"{train_time:,.1f}" if train_time is not None else "N/A",
        'Parameters': f"{model_results.get('params', 0):,}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))





