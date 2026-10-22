from django.urls import path
from assistant import views

app_name = 'assistant'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/analyze/', views.api_analyze_code, name='api_analyze'),
    path('api/documents/upload/', views.api_upload_document, name='api_upload_document'),
    path('api/documents/<int:doc_id>/delete/', views.api_delete_document, name='api_delete_document'),
    path('api/documents/load-samples/', views.api_load_samples, name='api_load_samples'),
    path('api/settings/', views.api_update_settings, name='api_update_settings'),
    path('api/status/', views.api_get_status, name='api_status'),
    path('api/search-preview/', views.api_search_preview, name='api_search_preview'),
]
