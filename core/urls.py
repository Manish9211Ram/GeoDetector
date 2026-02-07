from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze_map, name='analyze_map'),
    path('chat/', views.chat_bot, name='chat_bot'),
]
