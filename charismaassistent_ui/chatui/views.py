# --- Standardbibliotheken ---
import json
import requests

# --- Django Framework ---
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie




FASTAPI_BASE = getattr(settings, "FASTAPI_BASE", "http://localhost:8000")



def chat_page(request):
    return render(request, "chatui/chat.html")