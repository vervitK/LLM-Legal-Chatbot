from django.urls import path ,include
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LoginView

urlpatterns =[
    path(route='',view=views.home_page,name='index'),
    path(route='chatbot',view=views.chatbot,name='chatbot'),
    # path('accounts/login/', auth_views.LoginView.as_view(), name='login'),
    # path('accounts/login/', LoginView.as_view(), name='login'),
    path('accounts/login', include('django.contrib.auth.urls')),

    path(route='accounts/login/',view=views.login,name='login'),
    path(route='accounts/signup/',view=views.signup,name='signup'),
    path(route='logout',view=views.logout,name='logout'),
    path(route='privacy_policy',view=views.privacy_policy,name='privacy_policy'),
    
    path(route='term_use',view=views.term_use,name='term_use'),
    path(route='pricing',view=views.pricing,name='pricing'),
    path('send_message/', views.send_message_view, name='send_message'),

    # path('signup/', views.signup, name='signup'),
    # path('verify_email/<str:token>/', views.verify_email, name='verify_email'),

]

