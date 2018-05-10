from __future__ import unicode_literals

from django.db import models

from django.forms import Textarea
# Create your models here.

class Stock(models.Model):
    input_string = models.CharField(max_length=200)
    news_input = models.CharField(max_length=10000)
    formfield_overrides = {
        models.CharField: {'widget': Textarea()},
    }
