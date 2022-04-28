from django.urls import path
from . import views
from .views import PostReview, EditReview, DeleteReview

urlpatterns = [
    path('', PostReview.as_view(), name='review'),
    path('<int:pk>/edit/', EditReview.as_view(), name='edit-post'),
    path('<int:pk>/delete/', DeleteReview.as_view(), name='delete-post'),
    # path('get', views.chat, name='chat'),
]
