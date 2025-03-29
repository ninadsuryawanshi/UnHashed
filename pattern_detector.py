import re
import math
import itertools
import nltk
from nltk.corpus import words

nltk.download('words')
dictionary = set(words.words())
leetspeak_map = {
    '0': ['o'],
    '1': ['i', 'l'],
    '@': ['a'],
    '3': ['e'],
    '5': ['s'],
    '7': ['t'],
    '!': ['i']
}
CHAR_SET_SIZES = {
    'lowercase': 26,
    'uppercase': 26,
    'digits': 10,
    'special': 32
}
def extract_features(password):
    """Extract features from a password for analysis."""
    features = {}
    features['length'] = len(password)
    features['unique_chars'] = len(set(password))
    features['has_lowercase'] = bool(re.search(r'[a-z]', password))
    features['has_uppercase'] = bool(re.search(r'[A-Z]', password))
    features['has_digits'] = bool(re.search(r'\d', password))
    features['has_special'] = bool(re.search(r'[^a-zA-Z0-9]', password))
    char_set_size = (features['has_lowercase'] * CHAR_SET_SIZES['lowercase'] +
                     features['has_uppercase'] * CHAR_SET_SIZES['uppercase'] +
                     features['has_digits'] * CHAR_SET_SIZES['digits'] +
                     features['has_special'] * CHAR_SET_SIZES['special'])
    features['entropy'] = features['length'] * math.log2(max(char_set_size, 1)) if char_set_size > 0 else 0
    features['has_sequence'] = bool(re.search(r'(?:012|123|234|345|456|567|678|789|abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|aaa|bbb|ccc|ddd)', password.lower()))
    features['is_dictionary_word'] = password.lower() in dictionary
    features['is_leetspeak'] = check_leetspeak(password) 
    return features
def check_leetspeak(password):
    """Check if password is a leetspeak version of a dictionary word."""
    substitutions = []
    for char in password:
        if char in leetspeak_map:
            substitutions.append(leetspeak_map[char])
        else:
            substitutions.append([char.lower()])       
    for variant in itertools.product(*substitutions):
        candidate = ''.join(variant)
        if candidate in dictionary and candidate != password.lower():
            return True
    return False

def calculate_strength_score(features):
    score = 0 
    length_score = min(features['length'] * 2, 40)
    score += length_score        
    diversity = (features['has_lowercase'] + features['has_uppercase'] +
                 features['has_digits'] + features['has_special'])
    diversity_score = diversity * 7.5  
    score += diversity_score        
    entropy_score = min(features['entropy'] / 2, 30)  
    score += entropy_score       
    if features['is_dictionary_word']:
        score -= 40
    if features['is_leetspeak']:
        score -= 30
    if features['has_sequence']:
        score -= 20
    if features['length'] < 8:
        score -= 10   
    return max(0, min(100, score))
def get_reasoning(features):
    """Generate reasoning for password weakness."""
    reasons = []
    if features['is_dictionary_word']:
        reasons.append("Your password uses a common word.")
    if features['has_sequence']:
        reasons.append("Your password contains a predictable sequence.")
    if features['is_leetspeak']:
        reasons.append("Your password is a leetspeak version of a common word.")
    if features['length'] < 8:
        reasons.append("Your password is too short.")
    if features['entropy'] < 30:
        reasons.append("Your password lacks sufficient randomness.")
    return " ".join(reasons) if reasons else "No significant weaknesses detected."
def analyze_password(password):
    """Analyze a single password and return strength score and reasoning."""
    features = extract_features(password)
    strength_score = calculate_strength_score(features)
    reasoning = get_reasoning(features)
    
    return {
        'strength_score': strength_score,
        'reasoning': reasoning,
        'features': features
    }
if __name__ == "__main__":   
    test_passwords = ["password123", "P@ssw0rd", "Xy12!k9", "randomStr!ng"]   
    for password in test_passwords:
        result = analyze_password(password)
        print(f"\nPassword: {password}")
        print(f"Strength Score: {result['strength_score']:.2f}/100")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Features: {result['features']}")