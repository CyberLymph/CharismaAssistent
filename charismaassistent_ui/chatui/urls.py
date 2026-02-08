from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path("", views.chat_page, name="chat"),
    path("wizard/step-1/", views.wizard_step1, name="wizard_step1"),
    path("wizard/step-2/", views.wizard_step2, name="wizard_step2"),
]
            



if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)