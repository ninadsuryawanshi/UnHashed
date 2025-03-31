import torch
import torch.nn as nn
from gensim.models import FastText
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import getpass

# Define RNN Model (must match trainer)
class PasswordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        super(PasswordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2)  # Match trainer
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# Load Models and Data
def load_models(models_dir="models"):
    print(f"Loading models from {models_dir}...")
    
    try:
        with open(os.path.join(models_dir, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        
        rnn_model = PasswordRNN(len(vocab))
        rnn_model.load_state_dict(torch.load(os.path.join(models_dir, "rnn_model.pth")))
        rnn_model.eval()
        
        fasttext_model = FastText.load(os.path.join(models_dir, "fasttext_model.model"))
        
        with open(os.path.join(models_dir, "leaked_passwords.txt"), "r", encoding='utf-8') as f:
            leaked_passwords = set(line.strip().lower() for line in f)
        
        with open(os.path.join(models_dir, "breach_counts.pkl"), "rb") as f:
            breach_counts = pickle.load(f)
        
        print(f"Models loaded successfully:")
        print(f"- Vocabulary: {len(vocab)} characters")
        print(f"- FastText model: {fasttext_model.wv.vector_size} dimensions")
        print(f"- Leaked passwords: {len(leaked_passwords)} entries")
        print(f"- Breach counts: {len(breach_counts)} entries")
        
        return rnn_model, fasttext_model, vocab, leaked_passwords, breach_counts
    
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you've run the trainer script first.")
        exit(1)

# Calculate character-level entropy
def calculate_entropy(password):
    if not password:
        return 0
    
    char_count = {}
    for char in password:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    entropy = 0
    for count in char_count.values():
        probability = count / len(password)
        entropy -= probability * np.log2(probability)
    
    return entropy * len(password)  # Scale by password length

# Similarity Check
def get_similarity(password, fasttext_model, leaked_passwords, top_n=5):
    if not password:
        return 0.0, []
    
    # Convert password to character list
    chars = list(password.lower())
    
    # Check if all characters are in the fasttext model
    if not all(c in fasttext_model.wv for c in chars):
        return 0.0, []
    
    # Get password vector by averaging character vectors
    try:
        pw_vectors = [fasttext_model.wv[c] for c in chars]
        pw_vector = np.mean(pw_vectors, axis=0)
    except:
        return 0.0, []
    
    # Sample leaked passwords (for efficiency)
    sample_size = min(1000, len(leaked_passwords))
    leaked_sample = np.random.choice(list(leaked_passwords), sample_size, replace=False)
    
    # Calculate similarities
    similarities = []
    for leaked_pw in leaked_sample:
        try:
            leaked_chars = list(leaked_pw)
            if all(c in fasttext_model.wv for c in leaked_chars):
                leaked_vectors = [fasttext_model.wv[c] for c in leaked_chars]
                leaked_vector = np.mean(leaked_vectors, axis=0)
                sim = cosine_similarity([pw_vector], [leaked_vector])[0][0]
                similarities.append((leaked_pw, sim))
        except:
            continue
    
    # Sort by similarity and get top matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similarities = similarities[:top_n]
    
    return max([s[1] for s in similarities]) if similarities else 0.0, top_similarities

# Analyze Password
def analyze_password(password, rnn_model, fasttext_model, leaked_passwords, vocab, breach_counts, max_len=20):
    password_lower = password.lower()
    
    # Check for exact match
    exact_match = password_lower in leaked_passwords
    
    # Get actual breach count
    breach_count = breach_counts.get(password_lower, 0)
    
    # Calculate complexity metrics
    length_score = min(len(password) / 12, 1.0)  # Normalize to 0-1, 12+ chars gets 1.0
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    character_diversity = (has_upper + has_lower + has_digit + has_special) / 4
    
    # Calculate entropy
    entropy = calculate_entropy(password)
    entropy_score = min(entropy / 50, 1.0)  # Normalize to 0-1
    
    # RNN prediction for leaked patterns
    indices = [vocab.get(char.lower(), 0) for char in password[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        leak_confidence = rnn_model(input_tensor).item()
    
    # FastText similarity check
    similarity, similar_passwords = get_similarity(password, fasttext_model, leaked_passwords)
    
    # Password strength calculation
    # Invert leak confidence since higher leak confidence means weaker password
    strength_score = 0.4 * (1 - leak_confidence) + 0.3 * length_score + 0.2 * character_diversity + 0.1 * entropy_score
    strength_score = max(0, min(1, strength_score))  # Ensure 0-1 range
    
    # Reduce strength drastically if there's an exact match
    if exact_match:
        strength_score *= 0.2
    elif similarity > 0.8:
        strength_score *= 0.5
    
    # Calculate risk score (inverse of strength)
    risk_score = 1 - strength_score
    
    # Build comprehensive results
    results = {
        'exact_match': exact_match,
        'breach_count': breach_count,
        'length': len(password),
        'character_diversity': character_diversity,
        'entropy': entropy,
        'rnn_confidence': leak_confidence,
        'similarity_score': similarity,
        'similar_passwords': similar_passwords,
        'strength_score': strength_score,
        'risk_score': risk_score
    }
    
  
    reasoning = []
    if exact_match:
        reasoning.append(f"Your password was found in {breach_count} data breaches.")   
    if leak_confidence > 0.7:
        reasoning.append("Your password contains patterns commonly found in leaked passwords.")   
    if similarity > 0.8:
        reasoning.append("Your password is very similar to known leaked passwords.")   
    if len(password) < 8:
        reasoning.append("Your password is too short. A minimum of 8 characters is recommended.")  
    if character_diversity < 0.75:
        missing = []
        if not has_upper: missing.append("uppercase letters")
        if not has_lower: missing.append("lowercase letters")
        if not has_digit: missing.append("numbers")
        if not has_special: missing.append("special characters")
        reasoning.append(f"Your password lacks complexity. Consider adding {', '.join(missing)}.")
    
    if entropy < 30:
        reasoning.append("Your password has low entropy, making it potentially easier to guess.")
    
    if not reasoning:
        if strength_score > 0.8:
            reasoning.append("Your password looks strong and doesn't match known leaked patterns.")
        else:
            reasoning.append("Your password could be improved but doesn't match known leaked patterns.")   
    results['reasoning'] = reasoning   
    return results
def get_risk_level(score):
    if score < 0.2:
        return "Very Low", "🟢"
    elif score < 0.4:
        return "Low", "🟢"
    elif score < 0.6:
        return "Medium", "🟡"
    elif score < 0.8:
        return "High", "🟠"
    else:
        return "Very High", "🔴"
def display_results(result):
    risk_level, risk_emoji = get_risk_level(result['risk_score'])   
    print("\n" + "="*60)
    print(f"{risk_emoji} RISK LEVEL: {risk_level} ({result['risk_score']:.0%})")
    print("="*60)
    if result['exact_match']:
        print(f"⚠️  FOUND IN BREACHES: Yes (Count: {result['breach_count']})")
    else:
        print("✓  FOUND IN BREACHES: No")
    if result['similar_passwords']:
        print("\nSIMILAR LEAKED PASSWORDS:")
        for pw, score in result['similar_passwords']:
            if score > 0.5: 
                print(f"  - {pw} ({score:.0%} similar)")
    print("\nPASSWORD METRICS:")
    print(f"  Length: {result['length']} characters")
    print(f"  Character diversity: {result['character_diversity']:.0%}")
    print(f"  Entropy: {result['entropy']:.1f}")
    print(f"  Pattern match: {result['rnn_confidence']:.0%}")
    print(f"  Similarity to leaks: {result['similarity_score']:.0%}")
    print(f"  Overall strength: {result['strength_score']:.0%}")
    print("\nANALYSIS:")
    for reason in result['reasoning']:
        print(f"  • {reason}")   
    print("\nRECOMMENDATION:")
    if result['risk_score'] > 0.5:
        print("  You should change this password as soon as possible.")
    else:
        print("  This password is acceptable, but remember to use unique passwords for each site.")
    print("="*60)
def interactive_mode(rnn_model, fasttext_model, leaked_passwords, vocab, breach_counts):
    print("\n📊 PASSWORD LEAK DETECTOR 📊")
    print("Enter passwords to check against leak patterns (Ctrl+C to exit)")
    
    try:
        while True:
            print("\n" + "-"*60)
            password = getpass.getpass("Enter password to check: ")
            if not password:
                print("Please enter a password.")
                continue     
            result = analyze_password(password, rnn_model, fasttext_model, leaked_passwords, vocab, breach_counts)
            display_results(result)
    except KeyboardInterrupt:
        print("\nExiting password checker.")
if __name__ == "__main__":
    models_dir = "models"
    if not all(os.path.exists(os.path.join(models_dir, f)) for f in ["vocab.pkl", "rnn_model.pth", "fasttext_model.model", "leaked_passwords.txt", "breach_counts.pkl"]):
        print(f"Error: Pretrained models or data not found in '{models_dir}' directory.")
        print("Run train_leaked_model.py first to generate the necessary models.")
        exit(1)
    rnn_model, fasttext_model, vocab, leaked_passwords, breach_counts = load_models(models_dir)
    interactive_mode(rnn_model, fasttext_model, leaked_passwords, vocab, breach_counts)