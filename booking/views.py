from datetime import datetime
from django.contrib import messages
from django.shortcuts import render
from django.views.generic import ListView, CreateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from .forms import BookingForm
from .models import Room, Booking
from django.db.models import Q


# Create your views here.
def index(request):
    return render(request, 'booking/index.html')

def search(request):
    query = request.GET['query']

    if query < datetime.now().strftime('%Y-%m-%d'):
        messages.error(request, 'Enter a valid Date. Date cannot be in the past')
        return render(request, 'booking/index.html')
    else:
        rooms = Room.objects.filter(~Q(booking__date = query))
        params = {'rooms':rooms}
        return render(request, 'booking/booking_search.html', params)

class RoomListView(ListView):
    model = Room
    template_name = 'booking/room.html'
    context_object_name = 'rooms'

class BookingListView(LoginRequiredMixin, ListView):
    model = Booking
    template_name = 'booking/booking.html'
    context_object_name = 'bookings'
    ordering = ['date']

    def get_context_data(self,**kwargs):
        context = super(BookingListView,self).get_context_data(**kwargs)
        context['userBookings'] = Booking.objects.filter(booked_by = self.request.user)
        return context

class BookingCreateView(LoginRequiredMixin, CreateView):
    model = Booking
    form_class = BookingForm

    def form_valid(self, form):
        form.instance.booked_by = self.request.user
        return super().form_valid(form)
    
class BookingDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Booking
    success_url = '/bookings'

    def test_func(self):
        booking = self.get_object()
        if self.request.user == booking.booked_by:
            return True
        return False



        