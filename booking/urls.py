import imp
from django.urls import path
from . import views
from .views import RoomListView, BookingListView, BookingCreateView, BookingDeleteView

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search, name='search'),
    path('rooms/', RoomListView.as_view(), name='room-list'),
    path('bookings/', BookingListView.as_view(), name='booking-list'),
    path('bookings/new/', BookingCreateView.as_view(), name='booking-create'),
    path('rooms/<int:pk>/bookings/new/', BookingCreateView.as_view(), name='room-booking'),
    path('bookings/<int:pk>/delete/', BookingDeleteView.as_view(), name='booking-delete')
]