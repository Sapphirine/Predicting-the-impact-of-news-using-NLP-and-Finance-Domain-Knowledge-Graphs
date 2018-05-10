from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^stocks/$', views.stocks, name='stocks'),
    url(r'^news/$', views.news, name='news'),
    url(r'^news/newsinfo/$', views.newsinfo, name='newsinfo'),
    url(r'^stocks/stocksinfo/$', views.stocksinfo, name='stocksinfo'),
]
