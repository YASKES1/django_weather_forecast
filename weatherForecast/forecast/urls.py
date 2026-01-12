from django.urls import path
from . import views


urlpatterns = [
    path('', views.weatherView, name='Weather View'),
    
]