# Generated by Django 4.0.3 on 2022-04-21 12:58

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('booking', '0008_room_ac_room_wifi_alter_booking_booked_on'),
    ]

    operations = [
        migrations.AlterField(
            model_name='booking',
            name='booked_on',
            field=models.DateTimeField(default=datetime.datetime(2022, 4, 21, 12, 58, 4, 260878, tzinfo=utc)),
        ),
    ]
