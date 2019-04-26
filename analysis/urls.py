from django.urls import path
from . import views

urlpatterns=[
    path('', views.index, name='index'),
    path('xgboostanalyse',views.xgboostanalyse,name='xgboostanalyse'),
    path('randomforestanalyse',views.randomforestanalyse,name='randomforestanalyse'),
    path('adaboostanalyse',views.adaboostanalyse,name='adaboostanalyse'),
]