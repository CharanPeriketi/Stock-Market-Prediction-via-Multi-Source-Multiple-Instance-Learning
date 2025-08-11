from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("LoadDataset", views.LoadDataset, name="LoadDataset"),
	       path("TrainModels", views.TrainModels, name="TrainModels"),
	       path("Predict.html", views.Predict, name="Predict"),
               path("PredictAction", views.PredictAction, name="PredictAction"),	
	       path("MultiSource", views.MultiSource, name="MultiSource"),               
]
