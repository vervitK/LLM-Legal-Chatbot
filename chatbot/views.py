from django.contrib import auth, messages
# import openai
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone

from .models import Chat

# openai_api_key = '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'
# openai_api_key ='openai_api_key'
# openai.api_key = openai_api_key

def ask_openai(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                'role': 'user',
                'content': message
            }
        ],
        max_tokens=20,
        temperature=0.5
    )
    answer = response.choices[0].message.content.strip()
    return answer

# Create your views here.
@login_required
def chatbot(request):
    user_id = request.user.id

    chats = Chat.objects.filter(user=request.user,user_id=user_id)

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})


def home_page(request):
    return render(request=request,template_name='index.html')

def pricing(request):
    return render(request=request,template_name='pricing.html')
                  
def privacy_policy(request):
    return render(request=request,template_name='privacy_policy.html')


def term_use(request):
    return render(request=request,template_name='term_use.html')



#             auth.login(request, user)
#             return redirect('chatbot')
#         else:
#             error_message = 'Invalid username or password'
#             return render(request, 'login.html', {'error_message': error_message})
#     else:
#         return render(request, 'login.html')

# def login(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']
#         user = auth.authenticate(request, username=username, password=password)
#         if user is not None:
#             auth.login(request, user)
#             return redirect('chatbot')
#         else:
#             error_message = 'Invalid username or password'
#             return render(request, 'login.html', {'error_message': error_message})
#     else:
#         return render(request, 'login.html')

from django.shortcuts import render, redirect
from django.contrib import auth

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


# def register(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         email = request.POST['email']
#         password1 = request.POST['password1']
#         password2 = request.POST['password2']

#         if password1 == password2:
#             try:
#                 user = User.objects.create_user(username, email, password1)
#                 user.save()
#                 auth.login(request, user)
#                 return redirect('chatbot')
#             except:
#                 error_message = 'Error creating account'
#                 return render(request, 'signup.html', {'error_message': error_message})
#         else:
#             error_message = 'Password dont match'
#             return render(request, 'signup.html', {'error_message': error_message})
#     return render(request, 'signup.html')

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating account'
                return render(request, 'signup.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'signup.html', {'error_message': error_message})
    return render(request, 'signup.html')


# def signup(request):
#     if request.method == 'POST':
#         # Validate form data
#         username = request.POST.get('username', '')
#         email = request.POST.get('email', '')
#         password1 = request.POST.get('password1', '')
#         password2 = request.POST.get('password2', '')

#         if not username or not email or not password1 or not password2:
#             error_message = 'All fields are required.'
#             return render(request, 'signup.html', {'error_message': error_message})

#         if password1 != password2:
#             error_message = 'Passwords do not match.'
#             return render(request, 'signup.html', {'error_message': error_message})

#         try:
#             # Validate password against Django's password validation rules
#             validate_password(password1, user=User)

#             # Check if the username already exists
#             if User.objects.filter(username=username).exists():
#                 error_message = 'Username is already taken.'
#                 return render(request, 'signup.html', {'error_message': error_message})

#             user = User.objects.create_user(username=username, email=email, password=password1)
#             auth.login(repiquest, user)
#             return redirect('chatbot')
#         except ValidationError as e:
#             # Handle password validation errors
#             error_message = ', '.join(e.messages)
#             return render(request, 'signup.html', {'error_message': error_message})
#         except Exception as e:
#             # Handle other exceptions, if necessary
#             error_message = f'Error creating account: {str(e)}'
#             return render(request, 'signup.html', {'error_message': error_message})

#     return render(request, 'signup.html')


def logout(request):
    auth.logout(request)
    # return redirect('index.html')
    return render(request=request,template_name='index.html')



# # views.py
# from django.shortcuts import render, redirect
# from django.contrib import auth
# from django.contrib.auth.models import User
# from django.contrib.auth.password_validation import validate_password
# from django.core.exceptions import ValidationError
# from django.utils.crypto import get_random_string
# from django.core.mail import send_mail
# from django.conf import settings
# from .models import EmailVerification

# def signup(request):
#     if request.method == 'POST':
#         # Validate form data
#         username = request.POST.get('username', '')
#         email = request.POST.get('email', '')
#         password1 = request.POST.get('password1', '')
#         password2 = request.POST.get('password2', '')

#         if not username or not email or not password1 or not password2:
#             error_message = 'All fields are required.'
#             return render(request, 'signup.html', {'error_message': error_message})

#         if password1 != password2:
#             error_message = 'Passwords do not match.'
#             return render(request, 'signup.html', {'error_message': error_message})

#         try:
#             # Validate password against Django's password validation rules
#             validate_password(password1, user=User)

#             # Check if the username already exists
#             if User.objects.filter(username=username).exists():
#                 error_message = 'Username is already taken.'
#                 return render(request, 'signup.html', {'error_message': error_message})

#             # Create the user object but do not save it yet
#             user = User(username=username, email=email)
#             user.set_password(password1)
#             user.is_active = False  # Mark user as inactive until verified
#             user.save()

#             # Generate a unique verification token
#             verification_token = get_random_string(length=32)
#             EmailVerification.objects.create(user=user, token=verification_token)

#             # Send verification email
#             subject = 'Account Verification'
#             message = f'Click the following link to verify your account: {request.build_absolute_uri("/verify_email/" + verification_token)}'
#             from_email = settings.DEFAULT_FROM_EMAIL
#             recipient_list = [user.email]
#             send_mail(subject, message, from_email, recipient_list)

#             return render(request, 'verification_pending.html', {'email': user.email})

#         except ValidationError as e:
#             # Handle password validation errors
#             error_message = ', '.join(e.messages)
#             return render(request, 'signup.html', {'error_message': error_message})

#     return render(request, 'signup.html')

# def verify_email(request, token):
#     try:
#         verification = EmailVerification.objects.get(token=token)
#         user = verification.user
#         user.is_active = True
#         user.save()

#         verification.delete()

#         return render(request, 'verification_successful.html')
#     except EmailVerification.DoesNotExist:
#         error_message = 'Invalid verification link.'
#         return render(request, 'verification_failed.html', {'error_message': error_message})



def send_message_view(request):
    if request.method == 'POST':
        # Process the form data here
        name = request.POST.get('Name')
        email = request.POST.get('Email')
        phone = request.POST.get('Phone')
        message = request.POST.get('Message')

        # Perform any necessary actions with the form data
        # For example, you can send an email, save the data to a database, etc.

        # Send an email to the company with the form data
        subject = f"Contact Form Submission from {name}"
        from_email = '{email}'  # Replace with your email address or use email from the form
        to_email = 'company-email@example.com'  # Replace with the company's email address

        email_content = f"Name: {name}\nEmail: {email}\nPhone: {phone}\nMessage: {message}"

        send_mail(subject, email_content, from_email, [to_email])

        # Once you have processed the form data and sent the email, you can render a success page or redirect to another URL
        messages.success(request, "Your message has been sent successfully. We will get back to you soon!")
        return render(request, 'index.html')

    # If the request method is not POST, simply render the form page
    return render(request, 'index.html')