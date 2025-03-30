from django.urls import path
from .views import analyze_password

urlpatterns = [
    path('analyze-password/', analyze_password, name='analyze-password'),
]
