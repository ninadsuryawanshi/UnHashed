from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from analysis.ml_models.analyzer_main import analyze_password_comprehensive
from analysis.ml_models.suggest_passwords import suggest_improved_passwords  # Import the ML function

@csrf_exempt
def analyze_password(request):
    """API endpoint to analyze a password."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            password = data.get("password")

            if not password:
                return JsonResponse({"error": "Password is required"}, status=400)

            result = analyze_password_comprehensive(password)

            return JsonResponse(result, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Send a POST request with 'password' field."}, status=400)

@csrf_exempt
def password_suggestion_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            password = data.get("password")
            
            if not password:
                return JsonResponse({"error": "Password is required."}, status=400)
            
            # Generate password suggestions
            suggestions = suggest_improved_passwords(password)
            return JsonResponse({"suggestions": suggestions}, status=200)
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON input."}, status=400)
    
    return JsonResponse({"error": "Invalid request method."}, status=405)
