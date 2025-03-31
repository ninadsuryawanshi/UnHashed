import pandas as pd
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import FastText
import os
import pickle
import time

def preprocess_rockyou(file_path, sample_size=None):
    """Load and preprocess the RockYou dataset."""
    print("Loading and preprocessing RockYou dataset...")
    password_counter = Counter()
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
              password_counter[line.strip().lower()] += 1   
    print(f"Total unique passwords found: {len(password_counter)}" )
    if sample_size is not None:
        most_common_passwords = [pw for pw, _ in password_counter.most_common(sample_size)]
    else:
        most_common_passwords = list(password_counter.keys())
    print(f"Using {len(most_common_passwords)} passwords for training")
    tokenized = [list(pw) for pw in most_common_passwords]
    trigrams = [''.join(pw[i:i+3]) for pw in most_common_passwords for i in range(len(pw)-2) if len(pw) >= 3]
    pattern_freq = Counter(trigrams).most_common(20)
    print("\nTop 10 most common passwords:")
    for pw, count in password_counter.most_common(10):
        print(f"  '{pw}': {count} occurrences")   
    return most_common_passwords, tokenized, pattern_freq, password_counter
class PasswordDataset(Dataset):
    def __init__(self, passwords, vocab, max_len=20):
        self.passwords = passwords
        self.vocab = vocab
        self.max_len = max_len
        self.labels = [1] * len(passwords)  # All are "leaked"   
    def __len__(self):
        return len(self.passwords)   
    def __getitem__(self, idx):
        pw = self.passwords[idx]
        indices = [self.vocab.get(char, 0) for char in pw[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(1, dtype=torch.float)
class PasswordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        super(PasswordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2)  # Upgraded to LSTM
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)
def train_rnn(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0       
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Time: {time.time() - start_time:.2f}s")
        start_time = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_inputs, batch_labels in dataloader:
            outputs = model(batch_inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()        
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
def train_fasttext(tokenized_passwords):
    print("\nTraining FastText model...")
    start_time = time.time()
    model = FastText(sentences=tokenized_passwords, vector_size=64, window=3, min_count=1, workers=4, epochs=10)
    print(f"FastText training completed in {time.time() - start_time:.2f}s")
    return model
def main():
    rockyou_path = "rockyou.txt"  
    sample_size = 100000  
    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(rockyou_path):
        print(f"Error: {rockyou_path} not found.")
        exit(1)
    print(f"\n{'='*50}\nSTEP 1: PREPROCESSING\n{'='*50}")
    leaked_passwords, tokenized, pattern_freq, breach_counts = preprocess_rockyou(rockyou_path, sample_size)
    print(f"\nTop patterns: {pattern_freq[:5]}")
    print(f"\n{'='*50}\nSTEP 2: BUILDING VOCABULARY\n{'='*50}")
    all_chars = set(''.join(leaked_passwords)) | {'<PAD>'}
    vocab = {char: idx for idx, char in enumerate(all_chars)}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} characters")
    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)   
    with open(os.path.join(output_dir, "breach_counts.pkl"), "wb") as f:
        pickle.dump(dict(breach_counts), f)
    print(f"\n{'='*50}\nSTEP 3: TRAINING RNN MODEL\n{'='*50}")
    dataset = PasswordDataset(leaked_passwords, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    rnn_model = PasswordRNN(vocab_size)
    train_rnn(rnn_model, dataloader, epochs=3)
    torch.save(rnn_model.state_dict(), os.path.join(output_dir, "rnn_model.pth"))
    print(f"\n{'='*50}\nSTEP 4: TRAINING FASTTEXT MODEL\n{'='*50}")
    fasttext_model = train_fasttext(tokenized)
    fasttext_model.save(os.path.join(output_dir, "fasttext_model.model"))
    with open(os.path.join(output_dir, "leaked_passwords.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(leaked_passwords))
    print(f"\n{'='*50}\nTRAINING COMPLETE\n{'='*50}")
    print(f"Models and data saved to '{output_dir}' directory.")
    print(f"- Processed {len(leaked_passwords)} unique passwords")
    print(f"- Created vocabulary with {vocab_size} characters")
    print(f"- Saved breach counts for {len(breach_counts)} passwords")
    print(f"- Trained RNN model for password pattern recognition")
    print(f"- Trained FastText model for password similarity detection")

if __name__ == "__main__":
    main()