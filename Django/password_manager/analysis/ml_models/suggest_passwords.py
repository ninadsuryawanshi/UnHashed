import string
import random
import nltk
import os
import re
from nltk.corpus import words

try:
    WORD_LIST = words.words()  
except LookupError:
    print("Downloading NLTK words corpus...")
    nltk.download('words')
    WORD_LIST = words.words()

PASSPHRASE_WORDS = [word for word in WORD_LIST if len(word) >= 4 and len(word) <= 8]
LEETSPEAK_MAP = {
    'a': ['4', '@'],
    'b': ['8'],
    'e': ['3'],
    'i': ['1', '!'],
    'l': ['1'],
    'o': ['0'],
    's': ['5', '$'],
    't': ['7'],
    'z': ['2']
}

def apply_leetspeak(word, force_transform=False):
    """Apply leetspeak transformations to a word, with option to force at least one transform"""
    result = list(word.lower())
    num_transforms = min(3, len(result))
    positions = random.sample(range(len(result)), num_transforms)
    
    transformed = False
    for pos in positions:
        char = result[pos].lower()
        if char in LEETSPEAK_MAP:
            if force_transform or random.random() > 0.5:
                result[pos] = random.choice(LEETSPEAK_MAP[char])
                transformed = True
    
    # Ensure at least one transformation if force_transform is True
    if force_transform and not transformed:
        for i, char in enumerate(result):
            if char in LEETSPEAK_MAP:
                result[i] = random.choice(LEETSPEAK_MAP[char])
                break
    
    return ''.join(result)

def randomize_case(word):
    """Randomly change the case of characters in a word"""
    result = ""
    for char in word:
        if char.isalpha():
            result += char.upper() if random.random() > 0.7 else char.lower()
        else:
            result += char
    return result

def extract_patterns(password):
    """Identify common patterns in the password"""
    patterns = []
    if re.search(r'(?:012|123|234|345|456|567|678|789|abc|bcd|cde|def)', password.lower()):
        patterns.append("Contains common sequence")
    if re.search(r'(.)\1{2,}', password):
        patterns.append("Contains repeated characters")
    keyboard_patterns = ['qwerty', 'asdfg', 'zxcvb', '12345']
    if any(pattern in password.lower() for pattern in keyboard_patterns):
        patterns.append("Contains keyboard pattern")
    
    return patterns

def generate_passphrase():
    """Generate a meaningful passphrase from dictionary words"""
    # Select words that could form a simple, memorable phrase
    words = random.sample(PASSPHRASE_WORDS, 4)
    transformed_words = [
        words[0].capitalize(),           # e.g., "Sunny" (noun/adjective)
        words[1].lower(),                # e.g., "hill" (noun)
        apply_leetspeak(words[2]),       # e.g., "r0ad" (noun with leetspeak)
        randomize_case(words[3])         # e.g., "wAlk" (verb/action)
    ]
    
    # Use simple separators: digits or punctuation
    separators = [
        str(random.randint(0, 9)),       # e.g., "2"
        random.choice(string.punctuation),  # e.g., "!"
        str(random.randint(0, 9)) + random.choice(string.punctuation),  # e.g., "3#"
    ]
    
    result = transformed_words[0] + random.choice(separators) + \
             transformed_words[1] + random.choice(separators) + \
             transformed_words[2] + random.choice(separators) + \
             transformed_words[3]
    
    return result

def generate_enhanced_random_password(length=16):
    """Generate a completely random password with balanced character sets, no brackets"""
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    symbols = "!@#$%^&*+=_-|;:,.?~"  # Removed brackets: []{}<>
    
    # Ensure at least one character from each set
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(symbols)
    ]
    
    # Fill the rest randomly but with balanced distribution
    all_chars = lowercase + uppercase + digits + symbols
    password.extend(random.choices(all_chars, k=length-4))
    
    # Shuffle the password
    random.shuffle(password)
    return ''.join(password)

def improve_existing_password(password):
    """Improve an existing password by adding leetspeak and addressing weaknesses"""
    patterns = extract_patterns(password)
    
    # Start with the original password
    improved = list(password)
    
    # Address length issues
    if len(improved) < 12:
        improved.extend(random.choices(string.ascii_letters + string.digits + string.punctuation, 
                                      k=12-len(improved)))
    
    # Ensure character diversity
    if not any(c.isupper() for c in improved):
        idx = random.randint(0, len(improved)-1)
        if improved[idx].isalpha():
            improved[idx] = improved[idx].upper()
        else:
            improved.append(random.choice(string.ascii_uppercase))
    
    if not any(c.islower() for c in improved):
        idx = random.randint(0, len(improved)-1)
        if improved[idx].isalpha():
            improved[idx] = improved[idx].lower()
        else:
            improved.append(random.choice(string.ascii_lowercase))
    
    if not any(c.isdigit() for c in improved):
        idx = random.randint(0, len(improved))
        improved.insert(idx, random.choice(string.digits))
    
    if not any(c in string.punctuation for c in improved):
        idx = random.randint(0, len(improved))
        improved.insert(idx, random.choice("!@#$%^&*()_+-="))
    
    # Address common patterns
    if patterns:
        for pattern in patterns:
            if "sequence" in pattern.lower() or "repeated" in pattern.lower():
                for i in range(len(improved) - 2):
                    if improved[i] == improved[i+1] == improved[i+2]:
                        replacement = random.sample(string.ascii_letters + string.digits + "!@#$%^&*", 3)
                        improved[i:i+3] = replacement
                        break
    
    # Apply leetspeak with forced transformation
    improved = list(apply_leetspeak(''.join(improved), force_transform=True))
    
    return ''.join(improved)

def suggest_improved_passwords(password):
    """Generate three improved password suggestions"""
    suggestions = {}
    
    # Suggestion 1: Improve existing password with leetspeak
    improved = improve_existing_password(password)
    suggestions["Improved Original"] = improved
    
    # Suggestion 2: Generate a meaningful passphrase
    passphrase = generate_passphrase()
    suggestions["Secure Passphrase"] = passphrase
    
    # Suggestion 3: Generate a completely random password
    random_password = generate_enhanced_random_password(18)
    suggestions["Random Password"] = random_password
    
    return suggestions

def display_suggestions(suggestions):
    """Display password suggestions"""
    print("\n" + "="*60)
    print("PASSWORD SUGGESTIONS")
    print("="*60)
    
    for name, password in suggestions.items():
        print(f"\n{name}: {password}")

def main():
    try:
        password = input("\nEnter a password to improve: ")
        
        # Generate and display suggestions
        print("\nGenerating stronger password suggestions...")
        suggestions = suggest_improved_passwords(password)
        display_suggestions(suggestions)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

if __name__ == "__main__":
    main()