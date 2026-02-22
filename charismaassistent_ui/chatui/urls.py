from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path("", views.chat_page, name="chat"),
    path("api/analyze/", views.analyze_proxy, name="analyze_proxy"),
    path("wizard/step-1/", views.wizard_step1, name="wizard_step1"),
    path("wizard/step-2/", views.wizard_step2, name="wizard_step2"),
    path("api/analyze_stream/", views.analyze_stream_proxy, name="analyze_stream_proxy")
]
            



if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)