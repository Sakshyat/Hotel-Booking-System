# Generated by Django 4.0.3 on 2022-04-19 16:15

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('booking', '0005_booking_totalprice_alter_booking_booked_on'),
    ]

    operations = [
        migrations.AddField(
            model_name='booking',
            name='check_out_date',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='booking',
            name='booked_on',
            field=models.DateTimeField(default=datetime.datetime(2022, 4, 19, 16, 15, 56, 480986, tzinfo=utc)),
        ),
    ]
