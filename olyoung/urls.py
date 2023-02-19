#%%
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index ,name = 'index'),
    path('ver1', views.ver1, name = 'ver1'),
    path('ver1_1', views.ver1_1, name = 'ver1_1'),
    path('ver1_result', views.ver1_result, name = 'ver1_result'),
    path('ver2_result1', views.ver2_result1, name = 'ver2_result1'),
    path('ver2_result2', views.ver2_result2, name = 'ver2_result2'),
    path('ver2_result3', views.ver2_result3, name = 'ver2_result3')
]

#%%
# from django.contrib import admin
# from django.urls import path,include

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('', include('olyoung.urls')),
# ]