import datetime
from email.policy import default
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse

# Create your models here.
class Room(models.Model):
    room_type = models.CharField(max_length=200)
    capacity = models.IntegerField(default=2)
    beds = models.IntegerField()
    price = models.IntegerField()
    room_pic = models.ImageField(upload_to='profile_pics')

    availibility_choices = (
    ("Available", "Available"),
    ("No", "Not Available"),
    )

    ac = models.CharField(max_length=15, choices=availibility_choices, default='Available')
    wifi = models.CharField(max_length=15, choices=availibility_choices, default='Available')

    def __str__(self):
       return self.room_type

class Booking(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    date = models.DateField()
    check_out_date = models.DateField(blank=True, null=True)
    days = models.PositiveIntegerField(default=1)
    stay = models.PositiveIntegerField(blank=True, null=True)
    booked_on = models.DateTimeField(default=timezone.now())
    booked_by = models.ForeignKey(User, on_delete=models.CASCADE)
    totalPrice = models.PositiveIntegerField(blank=True, null=True)
    
    def get_absolute_url(self):
        return reverse('booking-list')

    def save(self, *args, **kwargs):
        self.totalPrice = self.room.price * self.days
        self.stay = self.days + 1
        self.check_out_date = self.date + datetime.timedelta(days=self.days)
        super(Booking, self).save(*args, **kwargs)

 