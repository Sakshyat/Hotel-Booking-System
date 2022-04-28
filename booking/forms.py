from datetime import date, datetime
from django import forms
from .models import Booking

class DateInput(forms.DateInput):
    input_type = 'date'

class BookingForm(forms.ModelForm):

    class Meta:
        model = Booking
        fields = ['room', 'date', 'days']
        widgets = {
            'date': DateInput()
        }    
    def clean(self):
        currentDate = datetime.today().strftime('%Y-%m-%d')
        bookingDate = self.cleaned_data['date'].strftime('%Y-%m-%d')
        
        if Booking.objects.filter(date=self.cleaned_data['date']).exists():
            raise forms.ValidationError('Booking not available for the given date')

        if currentDate > bookingDate:
            raise forms.ValidationError('Please enter a valid Date. Date cannot be in the past')
        
        return self.cleaned_data