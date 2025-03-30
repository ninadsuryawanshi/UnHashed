import os
import getpass
import torch
import numpy as np
from dataset_runner import load_models, analyze_password as analyze_leaks
from pattern_detector import analyze_password as analyze_patterns
from timetocrack import calculate_time_to_crack

def check_models_available():
    """Check if the required model files exist."""
    models_dir = "models"
    required_files = ["vocab.pkl", "rnn_model.pth", "fasttext_model.model", 
                      "leaked_passwords.txt", "breach_counts.pkl"]
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found.")
        return False
    
    missing_files = [f for f in required_files 
                     if not os.path.exists(os.path.join(models_dir, f))]
    
    if missing_files:
        print(f"Error: The following required model files are missing: {', '.join(missing_files)}")
        print("Run train_leaked_model.py first to generate the necessary models.")
        return False
        
    return True

def normalize_scores(leak_score, pattern_score, entropy_score):
    """Normalize all scores to a 0-100 scale with weights."""
    # Convert leak risk_score (0-1) to strength (0-100)
    leak_strength = (1 - leak_score) * 100
    
    # Pattern score is already 0-100
    
    # Convert entropy rating to a numeric score
    entropy_mapping = {
        "Very Weak": 10,
        "Weak": 30,
        "Moderate": 60,
        "Strong": 85,
        "Very Strong": 100
    }
    entropy_num_score = entropy_mapping.get(entropy_score, 50)
    
    # Weight the scores (adjust weights as needed)
    # Leak analysis is most important as it uses real-world data
    weights = {
        'leak': 0.5,
        'pattern': 0.3,
        'entropy': 0.2
    }
    
    final_score = (
        weights['leak'] * leak_strength +
        weights['pattern'] * pattern_score +
        weights['entropy'] * entropy_num_score
    )
    
    return min(max(final_score, 0), 100)  # Ensure score stays within 0-100

def get_strength_category(score):
    """Convert numeric score to strength category and emoji."""
    if score < 20:
        return "Very Weak", "🔴", "Change immediately!"
    elif score < 40:
        return "Weak", "🟠", "Needs significant improvement"
    elif score < 60:
        return "Moderate", "🟡", "Could be stronger"
    elif score < 80:
        return "Strong", "🟢", "Good password"
    else:
        return "Very Strong", "🟢", "Excellent password"

def display_comprehensive_results(password, leak_result, pattern_result, crack_result, final_score):
    """Display comprehensive results from all three analysis methods."""
    strength_category, emoji, recommendation = get_strength_category(final_score)
    
    print("\n" + "="*80)
    print(f"{emoji} PASSWORD STRENGTH ANALYSIS: {strength_category} ({final_score:.1f}/100)")
    print("="*80)
    
    # SECTION 1: SUMMARY
    print("\n📊 SUMMARY:")
    print(f"  • Overall Strength: {strength_category} ({final_score:.1f}/100)")
    print(f"  • Length: {len(password)} characters")
    print(f"  • Time to Crack: {crack_result['formatted_time']} (at 10^11 guesses/sec)")
    
    # Clear indication when password is NOT found in breaches
    if leak_result['exact_match']:
        print(f"  • ⚠️ FOUND IN DATA BREACHES: Yes (Count: {leak_result['breach_count']})")
    else:
        print(f"  • ✓ NOT FOUND IN DATA BREACHES")
        
    # Indicate if password matches common patterns
    if leak_result['rnn_confidence'] < 0.3:
        print(f"  • ✓ USES RARE PASSWORD PATTERN")
    
    # SECTION 2: SECURITY ASSESSMENT
    print("\n🔒 SECURITY ASSESSMENT:")
    
    # Combine all reasoning from different analyses
    all_reasons = leak_result['reasoning'] + [pattern_result['reasoning']]
    print("  Identified issues:")
    if any(reason and reason != "No significant weaknesses detected." for reason in all_reasons):
        for reason in all_reasons:
            if reason and reason != "No significant weaknesses detected.":
                print(f"  • {reason}")
    else:
        print("  • No significant weaknesses detected.")
    
    # SECTION 3: CHARACTER COMPOSITION
    print("\n🔤 CHARACTER COMPOSITION:")
    charset_details = crack_result['charset_details']
    for category, present in charset_details.items():
        print(f"  • {category.capitalize()}: {'✓' if present else '✗'}")
    
    # SECTION 4: DETAILED METRICS
    print("\n📈 DETAILED METRICS:")
    print(f"  • Leak analysis score: {(1-leak_result['risk_score'])*100:.1f}/100")
    print(f"  • Pattern analysis score: {pattern_result['strength_score']:.1f}/100")
    print(f"  • Entropy: {crack_result['entropy']:.1f} bits ({crack_result['strength_rating']})")
    
    # Similar passwords (if any)
    if leak_result['similar_passwords'] and any(score > 0.5 for _, score in leak_result['similar_passwords']):
        print("\n⚠️ SIMILAR LEAKED PASSWORDS:")
        for pw, score in leak_result['similar_passwords']:
            if score > 0.5:
                print(f"  • {pw} ({score:.0%} similar)")
    
    # SECTION 5: RECOMMENDATION
    print("\n💡 RECOMMENDATION:")
    print(f"  {recommendation}")
    
    if final_score < 60:
        print("\nTips to improve your password:")
        tips = []
        if len(password) < 12:
            tips.append("Make it longer (aim for at least 12 characters)")
        if not charset_details['uppercase']:
            tips.append("Add uppercase letters")
        if not charset_details['lowercase']:
            tips.append("Add lowercase letters")
        if not charset_details['digits']:
            tips.append("Add numbers")
        if not charset_details['symbols']:
            tips.append("Add special characters")
        if leak_result['exact_match'] or leak_result['similarity_score'] > 0.8:
            tips.append("Avoid common words and patterns")
        
        if not tips:
            tips.append("Use a random password generator")
            
        for i, tip in enumerate(tips, 1):
            print(f"  {i}. {tip}")
    
    print("="*80)

def analyze_password_comprehensive(password):
    """Run comprehensive password analysis using all three methods."""
    # Check if models are available for leak analysis
    models_available = check_models_available()
    
    # Run all analyses
    if models_available:
        try:
            # Load models for leak analysis
            rnn_model, fasttext_model, vocab, leaked_passwords, breach_counts = load_models()
            
            # Run leak analysis
            leak_result = analyze_leaks(password, rnn_model, fasttext_model, 
                                       leaked_passwords, vocab, breach_counts)
        except Exception as e:
            print(f"Warning: Leak analysis failed: {e}")
            # Create a default leak_result with neutral values
            leak_result = {
                'exact_match': False,
                'breach_count': 0,
                'risk_score': 0.5,
                'similarity_score': 0,
                'similar_passwords': [],
                'reasoning': ["Unable to perform leak analysis."],
                'rnn_confidence': 0.5  # Adding this field for consistency
            }
    else:
        print("Warning: Models not available. Skipping leak analysis.")
        # Create a default leak_result with neutral values
        leak_result = {
            'exact_match': False,
            'breach_count': 0,
            'risk_score': 0.5,
            'similarity_score': 0,
            'similar_passwords': [],
            'reasoning': ["Leak analysis skipped - models not available."],
            'rnn_confidence': 0.5  # Adding this field for consistency
        }
    
    # Run pattern analysis
    pattern_result = analyze_patterns(password)
    
    # Run time-to-crack analysis (using 10^11 as a standard for modern GPU)
    crack_result = calculate_time_to_crack(password, 10**11)
    
    # Calculate final score
    final_score = normalize_scores(
        leak_result['risk_score'], 
        pattern_result['strength_score'],
        crack_result['strength_rating']
    )
    
    return {
        'password': password,
        'leak_result': leak_result,
        'pattern_result': pattern_result,
        'crack_result': crack_result,
        'final_score': final_score
    }

def interactive_mode():
    """Run in interactive mode allowing multiple password checks."""
    try:
        while True:
            password = getpass.getpass("Enter password to analyze: ")
            if not password:
                print("Please enter a password.")
                continue
            
            results = analyze_password_comprehensive(password)
            display_comprehensive_results(
                results['password'],
                results['leak_result'],
                results['pattern_result'],
                results['crack_result'],
                results['final_score']
            )
            print("\n" + "-"*80)
    except KeyboardInterrupt:
        print("\nExiting password analyzer.")

def main():
    """Main function to run the password analyzer."""
    interactive_mode()

if __name__ == "__main__":
    main()