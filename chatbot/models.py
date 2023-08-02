from django.contrib.auth.models import User
from django.db import models


# Create your models here.
class Chat(models.Model):
    id = models.AutoField(primary_key=True)  # AutoField is typically used for IDs
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'
    

from django.contrib.auth.models import User
# models.py
from django.db import models
from django.utils import timezone


class EmailVerification(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100)
    created_at = models.DateTimeField(default=timezone.now)

