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
FASTAPI_URL = "http://localhost:8000/analyze"  # später docker-intern anpassen


def chat_page(request):
    return render(request, "chatui/chat.html")

def wizard_step1(request):
    """
    Step 1: Text input only. No backend call yet.
    We store text on the client (localStorage) for Step 2.
    """
    return render(request, "chatui/wizard_step1.html")

def wizard_step2(request):
    """
    Step 2: UI rendering of analysis result.
    For now: client-side mock JSON (until FastAPI/LLM is wired).
    """
    return render(request, "chatui/wizard_step2.html")

@require_POST
@csrf_exempt  
def analyze_proxy(request):
    payload = json.loads(request.body.decode("utf-8"))
    # expected: { "text": "...", "enable_compare": true/false }
    try:
      r = requests.post(FASTAPI_URL, json=payload, timeout=60)
      return JsonResponse(r.json(), status=r.status_code, safe=False)
    except requests.RequestException as e:
      return JsonResponse({"error": "backend_unavailable", "detail": str(e)}, status=503)