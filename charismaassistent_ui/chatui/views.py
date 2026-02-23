# --- Standard library ---
import json
import requests

# --- Django ---
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt


FASTAPI_BASE = getattr(settings, "FASTAPI_BASE", "http://localhost:8000")
FASTAPI_URL = f"{FASTAPI_BASE}/analyze"
FASTAPI_STREAM_URL = f"{FASTAPI_BASE}/analyze_stream"


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
    # expected payload: { "text": "...", "enable_compare": bool, "use_hybrid": bool }
    try:
        r = requests.post(FASTAPI_URL, json=payload, timeout=120)
        return JsonResponse(r.json(), status=r.status_code, safe=False)
    except requests.RequestException as e:
        return JsonResponse({"error": "backend_unavailable", "detail": str(e)}, status=503)


@require_POST
@csrf_exempt
def analyze_stream_proxy(request):
    """
    Proxies FastAPI NDJSON streaming endpoint 1:1 to the browser.
    Important: keep it StreamingHttpResponse (do NOT buffer).
    """
    payload = json.loads(request.body.decode("utf-8"))

    def gen():
        try:
            with requests.post(
                FASTAPI_STREAM_URL,
                json=payload,
                stream=True,
                timeout=300,
            ) as r:
                # If FastAPI returns an error, still return NDJSON error to the UI
                if r.status_code != 200:
                    try:
                        detail = r.text
                    except Exception:
                        detail = ""
                    err = {
                        "type": "error",
                        "message": f"backend_error_{r.status_code}: {detail[:300]}",
                    }
                    yield (json.dumps(err) + "\n").encode("utf-8")
                    return

                # Stream NDJSON as BYTES, robust to str/bytes from requests
                for line in r.iter_lines():
                    if not line:
                        continue

                    if isinstance(line, str):
                        line = line.encode("utf-8")

                    if not line.endswith(b"\n"):
                        line += b"\n"

                    yield line

        except requests.RequestException as e:
            err = {"type": "error", "message": f"backend_unavailable: {str(e)}"}
            yield (json.dumps(err) + "\n").encode("utf-8")

    resp = StreamingHttpResponse(gen(), content_type="application/x-ndjson")
    resp["Cache-Control"] = "no-cache"
    return resp