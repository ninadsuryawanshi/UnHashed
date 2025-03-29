import string
import math

def validate_input(password, guesses_per_second):
    """Validate user inputs"""
    if not password:
        raise ValueError("Password cannot be empty")
    if guesses_per_second <= 0:
        raise ValueError("Guesses per second must be a positive number")

def advanced_charset_classification(password):
    """Comprehensive character set analysis"""
    categories = {
        'lowercase': set(string.ascii_lowercase),
        'uppercase': set(string.ascii_uppercase),
        'digits': set(string.digits),
        'symbols': set(string.punctuation),
        'unicode': set(c for c in password if ord(c) > 127)
    }
    
    found_categories = {
        name: any(c in charset for c in password)
        for name, charset in categories.items()
    }
    
    return found_categories

def calculate_entropy(password, charset_size):
    """Calculate password entropy"""
    """ this could be done in more advanced approach,but kept it basic t avoid computational complexity. we may change that"""
    return len(password) * math.log2(charset_size)

def classify_charset(password):
    """Classify password charset and determine complexity"""
    has_lower = any(c in string.ascii_lowercase for c in password)
    has_upper = any(c in string.ascii_uppercase for c in password)
    has_digits = any(c in string.digits for c in password)
    has_symbols = any(c in string.punctuation for c in password)

    if has_digits and not (has_lower or has_upper or has_symbols):
        return "Numbers Only", 10
    elif (has_lower or has_upper) and not (has_digits or has_symbols):
        return "Alphabets Only", 26 if has_lower and not has_upper or has_upper and not has_lower else 52
    elif has_symbols and not (has_digits or has_lower or has_upper):
        return "Symbols Only", len(string.punctuation)
    else:
        return "Mixed Charset", (10 * has_digits) + (26 * has_lower) + (26 * has_upper) + (len(string.punctuation) * has_symbols)

def format_time(seconds):
    """Convert seconds to human-readable time format"""
    units = [
        (60 * 60 * 24 * 365, 'years'),
        (60 * 60 * 24, 'days'),
        (60 * 60, 'hours'),
        (60, 'minutes'),
        (1, 'seconds')
    ]
    
    for divisor, unit in units:
        if seconds >= divisor:
            value = seconds / divisor
            return f"{value:.2f} {unit}"
    
    return f"{seconds:.2f} seconds"

def rate_password_strength(entropy):
    """Provide a qualitative strength rating"""
    if entropy < 28:
        return "Very Weak"
    elif entropy < 36:
        return "Weak"
    elif entropy < 60:
        return "Moderate"
    elif entropy < 128:
        return "Strong"
    else:
        return "Very Strong"

def calculate_time_to_crack(password, guesses_per_second):
    """Estimate password cracking time with enhanced analysis"""
    validate_input(password, guesses_per_second)
    
    password_type, charset_size = classify_charset(password)
    search_space = charset_size ** len(password)
    time_to_crack = search_space / guesses_per_second
    
    # Calculate entropy
    entropy = calculate_entropy(password, charset_size)
    strength_rating = rate_password_strength(entropy)
    
    # Detailed charset analysis
    charset_details = advanced_charset_classification(password)
    
    return {
        'password': password,
        'length': len(password),
        'type': password_type,
        'charset_size': charset_size,
        'search_space': search_space,
        'time_to_crack': time_to_crack,
        'formatted_time': format_time(time_to_crack),
        'entropy': entropy,
        'strength_rating': strength_rating,
        'charset_details': charset_details
    }

# Example usage
def main():
    try:
        password = input("Enter the password: ")
        guesses_per_second = float(input("Enter guesses per second (e.g., 1e9 for high-end GPU): "))
        
        result = calculate_time_to_crack(password, guesses_per_second)
        
        print("\nPassword Analysis:")
        print(f"Password: {result['password']}")
        print(f"Length: {result['length']} characters")
        print(f"\nCracking Scenario:")
        print(f" - Type: {result['type']}")
        print(f" - Character Set Size: {result['charset_size']}")
        print(f" - Total Search Space: {result['search_space']:.2e} combinations")
        print(f"\nEstimated Crack Time:")
        print(f" - At {guesses_per_second:,.0f} guesses/second")
        print(f" - Theoretical Time: {result['formatted_time']}")
        
        print("\nStrength Assessment:")
        print(f" - Entropy: {result['entropy']:.2f} bits")
        print(f" - Strength Rating: {result['strength_rating']}")
        
        print("\nCharacter Set Composition:")
        for category, present in result['charset_details'].items():
            print(f" - {category.capitalize()}: {'✓' if present else '✗'}")
    
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()