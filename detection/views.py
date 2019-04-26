from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from sklearn.externals import joblib
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot

matplotlib.use('Agg')
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split


def detect(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location='usecases')
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        model_value = request.POST.get('model')

        if model_value == '1':
            filename = 'finalized_modelxgboost.sav'
        elif model_value == '2':
            filename = 'finalized_modelrandomforest.sav'
        elif model_value == '3':
            filename = 'finalized_modeladaboost.sav'
        else:
            return render(request, 'detection/index.html')
        loaded_model = joblib.load(filename)
        cc1 = pd.read_csv('usecases/'+uploaded_file_url)
        # test the model
        cc = cc1.iloc[:, 1:30].columns
        ccdata = cc1[cc]

        # the below code to be used for single line prediction
        y_pred = loaded_model.predict(ccdata)
        return render(request, 'detection/index.html', {
            'uploaded_file_url': uploaded_file_url,
            'y_pred': y_pred[0]
        })
    return render(request, 'detection/index.html')

