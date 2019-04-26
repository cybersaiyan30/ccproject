from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from django.http import HttpResponseRedirect

# Create your views here.
def index(request):
    return render(request,'analysis/index.html')


def xgboostanalyse(request):
    data = pd.read_csv('creditcard.csv')
    X = data.iloc[:, 1:30].columns
    y = data['Class']
    X = data[X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    filename = 'finalized_modelxgboost.sav'
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(X_test)
    a = (metrics.accuracy_score(y_test, y_pred))
    cm = (metrics.confusion_matrix(y_test, y_pred))
    df_cm=pd.DataFrame(cm)
    svm = sns.heatmap(df_cm, annot=True, cmap='coolwarm', linecolor='white', linewidths=1)

    figure = svm.get_figure()
    figure.savefig('analysis/static/analysis/images/xgboostconf.png', dpi=400)
    context={
        'accuracy':a,
    }
    figure.clf()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    matplotlib.pyplot.plot(fpr, tpr, label='data, auc=' + str(auc))
    matplotlib.pyplot.legend(loc=4)
    matplotlib.pyplot.savefig('analysis/static/analysis/images/xgboostgraph.png')
    matplotlib.pyplot.clf()
    return render(request,'analysis/xgboostanalyseht.html',context)

def randomforestanalyse(request):
    data = pd.read_csv('creditcard.csv')
    X = data.iloc[:, 1:30].columns
    y = data['Class']
    X = data[X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    filename = 'finalized_modelrandomforest.sav'
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(X_test)
    a = (metrics.accuracy_score(y_test, y_pred))
    cm = (metrics.confusion_matrix(y_test, y_pred))
    df_cm=pd.DataFrame(cm)
    svm = sns.heatmap(df_cm, annot=True, cmap='coolwarm', linecolor='white', linewidths=1)

    figure = svm.get_figure()
    figure.savefig('analysis/static/analysis/images/randomforestconf.png', dpi=400)
    context={
        'accuracy':a,
    }
    figure.clf()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    matplotlib.pyplot.plot(fpr, tpr, label='data, auc=' + str(auc))
    matplotlib.pyplot.legend(loc=4)
    matplotlib.pyplot.savefig('analysis/static/analysis/images/randomforestgraph.png')
    matplotlib.pyplot.clf()
    return render(request,'analysis/randomforestht.html',context)

def adaboostanalyse(request):
    data = pd.read_csv('creditcard.csv')
    X = data.iloc[:, 1:30].columns
    y = data['Class']
    X = data[X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    filename = 'finalized_modeladaboost.sav'
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(X_test)
    a = (metrics.accuracy_score(y_test, y_pred))
    cm = (metrics.confusion_matrix(y_test, y_pred))
    df_cm=pd.DataFrame(cm)
    svm = sns.heatmap(df_cm, annot=True, cmap='coolwarm', linecolor='white', linewidths=1)

    figure = svm.get_figure()
    figure.savefig('analysis/static/analysis/images/adaboostconf.png', dpi=400)
    context={
        'accuracy':a,
    }
    figure.clf()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    matplotlib.pyplot.plot(fpr, tpr, label='data, auc=' + str(auc))
    matplotlib.pyplot.legend(loc=4)
    matplotlib.pyplot.savefig('analysis/static/analysis/images/adaboostgraph.png')
    matplotlib.pyplot.clf()
    return render(request,'analysis/adaboostht.html',context)