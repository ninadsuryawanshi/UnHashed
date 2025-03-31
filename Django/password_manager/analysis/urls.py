from django.urls import path
from .views import analyze_password, password_suggestion_view

urlpatterns = [
    path('analyze-password/', analyze_password, name='analyze-password'),
    path('suggestions/',password_suggestion_view, name ='suggestions' ),
]
