from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
from django.core.files.storage import FileSystemStorage
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import os
import re
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

import spacy
from nltk.stem import WordNetLemmatizer
import nltk
from lib.sentence2vec import Sentence2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from xgboost import XGBClassifier

global uname, graph, xgb_cls
model = Sentence2Vec('data/job_titles.model')
nlp = spacy.load("en_core_web_sm")  
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def extract_events_nlp(sentence):
    doc = nlp(sentence)
    events = []
    for token in doc:
        # Identify potential predicates (verbs)
        if token.pos_ == "VERB":
            event = {"predicate": lemmatizer.lemmatize(token.text, 'v'), "arguments": {}}
            # Find subject (nsubj, nsubjpass)
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    event["arguments"]["subject"] = child.text
                    break
            # Find direct object (dobj)
            for child in token.children:
                if child.dep_ == "dobj":
                    event["arguments"]["object"] = child.text
                    break
            # Find indirect object (iobj)
            for child in token.children:
                if child.dep_ == "iobj":
                    event["arguments"]["indirect_object"] = child.text
                    break
            # Find prepositional phrases (prep) - often indicating location, time, etc.
            for child in token.children:
                if child.dep_ == "prep":
                    prep_phrase = child.text
                    pobj = None
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            pobj = grandchild.text
                            break
                    if pobj:
                        event["arguments"][f"prep_{prep_phrase}"] = pobj
            # Consider adverbs (advmod) for manner, time, etc.
            for child in token.children:
                if child.dep_ == "advmod":
                    event["arguments"]["adverbial_modifier"] = child.text
            events.append(event)
    return events

def get_all_values(data):
    all_values = []
    if isinstance(data, dict):
        for value in data.values():
            all_values.extend(get_all_values(value))
    elif isinstance(data, list):
        for item in data:
            all_values.extend(get_all_values(item))
    else:
        all_values.append(data)
    return all_values

def getVector(data):
    data = ' '.join(data)
    data = model.get_vector(data)
    return data

def getSentiment(data):
    sentiment = sia.polarity_scores(data)
    negative = sentiment['neg']
    positive = sentiment['pos']
    neutral = sentiment['neu']
    compound = sentiment['compound']
    return compound, negative, positive, neutral

X = []
Y = []

news = pd.read_csv("Dataset/Combined_News_DJIA.csv")
stock = pd.read_csv("Dataset/upload_DJIA_table.csv")
combined = news.merge(stock, how = 'inner', on = 'Date')

if os.path.exists("model/X.npy"):
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
else:
    X.clear()
    Y.clear()
    headline = []
    for row in range(0,len(combined.index)):
        headline.append(" ".join(str(x) for x in combined.iloc[row, 2:27]))
    clean_headline = []
    for i in range (0, len(headline)):
        clean_headline.append(re.sub("b[(')]", '', headline[i]))
        clean_headline[i] = re.sub('b[(")]', '', clean_headline[i])
        clean_headline[i] = re.sub("\'", '', clean_headline[i])
    Y = combined['Label'].ravel()
    for i in range(len(clean_headline)):
        ch = clean_headline[i].lower().strip()
        compound, negative, positive, neutral = getSentiment(ch)
        events = extract_events_nlp(ch)
        vector = getVector(get_all_values(events))
        df = combined.iloc[i]
        open_value = df['Open'].ravel()[0]
        high_value = df['High'].ravel()[0]
        low_value = df['Low'].ravel()[0]
        close_value = df['Close'].ravel()[0]
        volume_value = df['Volume'].ravel()[0]
        adj_value = df['Adj Close'].ravel()[0]
        vector = vector.tolist()
        vector.append(compound)
        vector.append(negative)
        vector.append(positive)
        vector.append(neutral)
        vector.append(open_value)
        vector.append(high_value)
        vector.append(low_value)
        vector.append(close_value)
        vector.append(volume_value)
        vector.append(adj_value)
        vector = np.asarray(vector)
        X.append(vector)
        print(str(i)+" "+str(vector)+" "+str(vector.shape))
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save("model/X", X)
    np.save("model/Y", Y)

accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    global graph
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data1.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

svm_cls = svm.SVC(kernel='sigmoid')
svm_cls.fit(X_train[:,104:110], y_train)
predict = svm_cls.predict(X_test[:,104:110])
calculateMetrics("Existing SVM", y_test, predict)

svm_cls = svm.SVC()
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("Propose Multi Instance", y_test, predict)

data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
xgb_cls = XGBClassifier()
xgb_cls.fit(X_train, y_train)
predict = xgb_cls.predict(X_test)
conf_matrix = confusion_matrix(y_test, predict)
calculateMetrics("Extension XGBoost", y_test, predict)

def TrainModels(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, conf_matrix
        labels = ['Decline', 'Rise']
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
        output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['Existing SVM', 'Propose Multi-Source Multi-Instance', 'Extension Multi-Source XGBoost']
        for i in range(len(algorithms)):
            output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['Existing SVM','Accuracy',accuracy[0]],['Existing SVM','Precision',precision[0]],['Existing SVM','Recall',recall[0]],['Existing SVM','FSCORE',fscore[0]],
                           ['Propose M-MI','Accuracy',accuracy[1]],['Propose M-MI','Precision',precision[1]],['Propose M-MI','Recall',recall[1]],['Propose M-MI','FSCORE',fscore[1]],
                           ['Extension M-XGBoost','Accuracy',accuracy[2]],['Extension M-XGBoost','Precision',precision[2]],['Extension M-XGBoost','Recall',recall[2]],['Extension M-XGBoost','FSCORE',fscore[2]],
                          ],columns=['Parameters','Algorithms','Value'])

        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])    
        df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def prediction(msg, model):
    embeddings = bert.encode([msg], convert_to_tensor=True)#apply bert on news data to start embedding
    msg = embeddings.numpy()
    predict = model.predict(msg)
    output = "<font size=4 color=green>HAM</option>"
    if predict[0] == 1:
        output = "<font size=4 color=red>SPAM</option>"
    return output    

def PredictAction(request):
    if request.method == 'POST':
        global scaler, xgb_cls
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("StockApp/static/"+fname):
            os.remove("StockApp/static/"+fname)
        with open("StockApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv("StockApp/static/"+fname)
        headline = []
        for row in range(0,len(testData.index)):
            headline.append(" ".join(str(x) for x in testData.iloc[row, 1:26]))
        clean_headline = []
        for i in range (0, len(headline)):
            clean_headline.append(re.sub("b[(')]", '', headline[i]))
            clean_headline[i] = re.sub('b[(")]', '', clean_headline[i])
            clean_headline[i] = re.sub("\'", '', clean_headline[i])
        test = []
        for i in range(len(clean_headline)):
            ch = clean_headline[i].lower().strip()
            compound, negative, positive, neutral = getSentiment(ch)
            events = extract_events_nlp(ch)
            vector = getVector(get_all_values(events))
            df = testData.iloc[i]
            open_value = df['Open'].ravel()[0]
            high_value = df['High'].ravel()[0]
            low_value = df['Low'].ravel()[0]
            close_value = df['Close'].ravel()[0]
            volume_value = df['Volume'].ravel()[0]
            adj_value = df['Adj Close'].ravel()[0]
            vector = vector.tolist()
            vector.append(compound)
            vector.append(negative)
            vector.append(positive)
            vector.append(neutral)
            vector.append(open_value)
            vector.append(high_value)
            vector.append(low_value)
            vector.append(close_value)
            vector.append(volume_value)
            vector.append(adj_value)
            vector = np.asarray(vector)
            test.append(vector)
        test = np.asarray(test)
        test = scaler.transform(test)
        predict = xgb_cls.predict(test)
        labels = ['Decline', 'Rise']
        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Data</th>'
        output += '<th><font size="3" color="black">Stock Prediction</th></tr>'
        for i in range(len(predict)):
            output += '<tr><td><font size="3" color="black">'+str(test[i])+'</td>'
            if predict[i] == 1:
                output += '<td><font size="3" color="green">'+str(labels[predict[i]])+'</font></td></tr>'
            else:
                output += '<td><font size="3" color="red">'+str(labels[predict[i]])+'</font></td></tr>'
        output+= "</table></br></br></br></br>"    
        context= {'data':output}
        return render(request, 'UserScreen.html', context)        

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        

def MultiSource(request):
    if request.method == 'GET':
        global X, Y, combined
        context= {'data':str(X)}
        return render(request, 'UserScreen.html', context)   

def LoadDataset(request):
    if request.method == 'GET':
        global X, Y, combined
        output = "Total Records found in Dataset = "+str(X.shape[0])+"<br/>"
        output += "<br/>Labels found in Dataset = Decline & Rise<br/>"
        output += "<br/>Dataset Train & Test Split Details<br/>"
        output += "80% records using to train Algorithms : "+str(X_train.shape[0])+"<br/>"
        output += "20% records using to test Algorithms : "+str(X_test.shape[0])+"<br/><br/>"
        columns = combined.columns
        combined = combined.values
        output+='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(combined)):
            output += '<tr>'
            for j in range(len(combined[i])):
                output += '<td><font size="3" color="black">'+str(combined[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)      

