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

# def ask_openai(message):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 'role': 'user',
#                 'content': message
#             }
#         ],
#         max_tokens=20,
#         temperature=0.5
#     )
#     answer = response.choices[0].message.content.strip()
#     return answer

from pathlib import Path

# Example: Trying different encodings
encodings_to_try = ['utf-8']
questions_dir = Path('home/ubuntu/project/chatbot/skyscanner')
for file_path in questions_dir.glob('*.txt'):
    for encoding in encodings_to_try:
        try:
            with file_path.open('r', encoding=encoding) as text_file:
                content = text_file.read()
                # Process the file content as needed
                break  # Stop trying encodings if successful
        except UnicodeDecodeError:
            pass



from pathlib import Path
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, pipeline
from functools import lru_cache
from functools import cached_property


class Chatbot:

    global DEVICE
    global tokenizer
    model_name_or_path = "TheBloke/Nous-Hermes-13B-GPTQ"
    model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEFAULT_TEMPLATE = """
    ### Instruction: You're a lawyer support agent that is talking to a clients. Use only the chat history and the following information
    {context}
    to answer in a helpful manner to the question. If you don't know the answer - say that you don't know.
    Keep your replies short, compassionate and informative.
    {chat_history}
    ### Input: {question}
    ### Response:
    """.strip()


    def __init__(
        self,
        prompt_template: str = DEFAULT_TEMPLATE,
        verbose: bool = False,):
        self.prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=prompt_template,)
        self.documents_dir = "home/ubuntu/project/chatbot/skyscanner"
        self.model_name_or_path = "TheBloke/Nous-Hermes-13B-GPTQ"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.text_pipeline = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2048,
                temperature=0,
                top_p=0.95,
                repetition_penalty=1.15,
                generation_config=self.generation_config,
                streamer=self.streamer,
                batch_size=1,
            )
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="embaas/sentence-transformers-multilingual-e5-base",
            model_kwargs={"device": DEVICE},
        )
        self.chain = self._create_chain(self.text_pipeline, self.prompt, verbose)
        self.db = self._embed_data(self.documents_dir, self.embeddings)
        # self.DEVICE = self.

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    @cached_property
    def model(self):
        return AutoGPTQForCausalLM.from_quantized(
            self.model_name_or_path,
            model_basename=self.model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device=DEVICE,
        )

    @cached_property
    def generation_config(self):
        return GenerationConfig.from_pretrained(self.model_name_or_path)

    @cached_property
    def streamer(self):
        return TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, use_multiprocessing=False
        )

    def _create_chain(
        self,
        text_pipeline: HuggingFacePipeline,
        prompt: PromptTemplate,
        verbose: bool = False,
    ):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            human_prefix="### Input",
            ai_prefix="### Response",
            input_key="question",
            output_key="output_text",
            return_messages=False,
        )
        return load_qa_chain(
            text_pipeline,
            chain_type="stuff",
            prompt=prompt,
            memory=memory,
            verbose=verbose,
        )

    def _embed_data(self, documents_dir: Path, embeddings: HuggingFaceEmbeddings) -> Chroma:
        loader = DirectoryLoader(documents_dir, glob="**/*txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return Chroma.from_documents(texts, embeddings)

    def __call__(self, user_input: str) -> str:
        docs = self.db.similarity_search(user_input)
        return self.chain.run({"input_documents": docs, "question": user_input})

# Create an instance of Chatbot


response_ai = Chatbot()

def ask_openai(message):
    ans = response_ai('You: ' + message)
    return ans




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
