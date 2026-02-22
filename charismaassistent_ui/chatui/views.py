# --- Standardbibliotheken ---
import json
import requests

# --- Django Framework ---
from django.conf import settings
from django.http import JsonResponse , StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie




FASTAPI_BASE = getattr(settings, "FASTAPI_BASE", "http://localhost:8000")
FASTAPI_URL = "http://localhost:8000/analyze"  # später docker-intern anpassen
FASTAPI_URL_STREAM = "http://localhost:8000/analyze_stream"  # FastAPI streaming endpoint


def chat_page(request):
    return render(request, "chatui/chat.html")


def wizard_step1(request):
    return render(request, "chatui/wizard_step1.html")


def wizard_step2(request):
    return render(request, "chatui/wizard_step2.html")


@require_POST
@csrf_exempt
def analyze_proxy(request):
    payload = json.loads(request.body.decode("utf-8"))
    try:
        r = requests.post(FASTAPI_URL, json=payload, timeout=120)
        return JsonResponse(r.json(), status=r.status_code, safe=False)
    except requests.RequestException as e:
        return JsonResponse({"error": "backend_unavailable", "detail": str(e)}, status=503)


@require_POST
@csrf_exempt
def analyze_stream_proxy(request):
    """
    Proxies FastAPI /analyze_stream NDJSON stream to the browser.
    Frontend expects `application/x-ndjson` (one JSON object per line).
    """
    payload = json.loads(request.body.decode("utf-8"))

    try:
        upstream = requests.post(
            FASTAPI_URL_STREAM,
            json=payload,
            stream=True,       # IMPORTANT: keep stream
            timeout=300,
        )

        # Pass through status if upstream fails quickly
        if upstream.status_code >= 400:
            try:
                err_text = upstream.text
            except Exception:
                err_text = "Upstream error"
            return JsonResponse(
                {"error": "upstream_error", "status": upstream.status_code, "detail": err_text[:500]},
                status=502,
            )

        def gen():
            # iter_lines keeps line boundaries; decode_unicode gives str
            for line in upstream.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                line = line.strip()
                if not line:
                    continue
                # Ensure every line ends with newline for NDJSON parsing
                yield (line + "\n")

        resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson; charset=utf-8")
        # Disable buffering in common reverse proxies
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"  # nginx
        return resp

    except requests.RequestException as e:
        return JsonResponse({"error": "backend_unavailable", "detail": str(e)}, status=503)