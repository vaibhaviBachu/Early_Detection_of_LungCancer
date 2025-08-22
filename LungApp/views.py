from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pymysql
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import matplotlib.pyplot as plt #use to visualize dataset vallues
import seaborn as sns
import io
import base64
import random
import smtplib
from datetime import date

global uname, utype, otp
global X_train, X_test, y_train, y_test, X, Y, dataset, rf, scaler, le


dataset = pd.read_excel("Dataset/Dataset.xlsx")
dataset.drop(['index', 'Patient Id'], axis = 1,inplace=True)
le = LabelEncoder()
dataset['Level'] = pd.Series(le.fit_transform(dataset['Level'].astype(str)))#encode all str columns to numeric
dataset.fillna(0, inplace = True)
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

rf = RandomForestClassifier(max_depth=2)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
p = precision_score(y_test, predict,average='macro') * 100
r = recall_score(y_test, predict,average='macro') * 100
f = f1_score(y_test, predict,average='macro') * 100
a = accuracy_score(y_test,predict)*100

def getGraph():
    dataset = pd.read_excel("Dataset/Dataset.xlsx")
    fig, axs = plt.subplots(2,2,figsize=(10, 6))
    data = dataset.groupby("Gender")["Chest Pain"].size().reset_index()
    data.Gender.replace([1.0, 2.0], ['Male', 'Female'], inplace=True)
    axs[0,0].pie(data["Chest Pain"], labels = data["Gender"], autopct="%1.1f%%")
    axs[0,0].set_title("Count of Chest Pain Gender Wise")

    data = dataset.groupby(["Gender", "Smoking"])['Alcohol use'].sum().reset_index()
    data.Gender.replace([1.0, 2.0], ['Male', 'Female'], inplace=True)
    sns.pointplot(data=data, x="Smoking", y="Alcohol use", hue="Gender", ax=axs[0,1])
    axs[0,1].set_title("Gender Wise Smoking & Alcohol Usage Graph")

    data = dataset.groupby(["Gender"])['OccuPational Hazards'].sum().reset_index()
    data.Gender.replace([1.0, 2.0], ['Male', 'Female'], inplace=True)
    sns.barplot(data=data, x="Gender", y='OccuPational Hazards', ax=axs[1,0])
    axs[1,0].set_title("Count of Patients Occupational Hazard Gender Wise Graph")

    data = dataset.groupby(["Snoring", "Level"])['Dry Cough'].count().reset_index()
    sns.pointplot(data=data, x="Snoring", y="Dry Cough", hue="Level", ax=axs[1,1])
    axs[1,1].set_title("Snoring by Dry Cough Graph")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()     
    return img_b64

def DatasetVisualize(request):
    if request.method == 'GET':
        img_b64 = getGraph()
        context= {'data':"", 'img': img_b64}
        return render(request, 'PatientScreen.html', context)        

def PredictDisease(request):
    if request.method == 'GET':
       return render(request, 'PredictDisease.html', {})

def AnalysePatient(request):
    if request.method == 'GET':
        output = '<table border=1><tr>'
        output+='<td><font size="" color="black">Patient Name</td>'
        output+='<td><font size="" color="black">Input Values</td>'
        output+='<td><font size="" color="black">Predicted Stage</td>'
        output+='<td><font size="" color="black">Date</td></tr>'
        rank = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'lungdisease',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM patients")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr>'
                output+='<td><font size="" color="black">'+str(row[0])+'</td>'
                output+='<td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+str(row[3])+'</td></tr>'
                rank.append(row[2])
        output += "</table><br/><br/><br/>"
        unique, count = np.unique(np.asarray(rank), return_counts=True)
        plt.pie(count,labels=unique,autopct='%1.1f%%')
        plt.title('Patients Lung Cancer Graph')
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'DoctorScreen.html', context)    

def PredictDiseaseAction(request):
    if request.method == 'POST':
        global uname, rf, scaler
        age = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        pollution = request.POST.get('t3', False)
        alcohol = request.POST.get('t4', False)
        dust = request.POST.get('t5', False)
        hazard = request.POST.get('t6', False)
        genetic = request.POST.get('t7', False)
        chronic = request.POST.get('t8', False)
        diet = request.POST.get('t9', False)
        obesity = request.POST.get('t10', False)
        smoking = request.POST.get('t11', False)
        smoker = request.POST.get('t12', False)
        chest = request.POST.get('t13', False)
        blood = request.POST.get('t14', False)
        fatigue = request.POST.get('t15', False)
        weight = request.POST.get('t16', False)
        breathe = request.POST.get('t17', False)
        wheezing = request.POST.get('t18', False)
        difficulty = request.POST.get('t19', False)
        finger = request.POST.get('t20', False)
        cold = request.POST.get('t21', False)
        cough = request.POST.get('t22', False)
        snoring = request.POST.get('t23', False)
        input_values = age+","+gender+","+pollution+","+alcohol+","+dust+","+hazard+","+genetic+","+chronic+","+diet+","+obesity+","+smoking+","+smoker+","+chest+","+blood+","+fatigue+","+weight+","+breathe+","+wheezing+","+difficulty+","+finger+","+cold+","+cough+","+snoring

        data = []
        data.append([float(age), float(gender), float(pollution), float(alcohol), float(dust), float(hazard), float(genetic), float(chronic), float(diet), float(obesity),
                     float(smoking), float(smoker), float(chest), float(blood), float(fatigue), float(weight), float(breathe), float(wheezing), float(difficulty),
                     float(finger), float(cold), float(cough), float(snoring)])
        data = np.asarray(data)
        print(data.shape)
        data = scaler.transform(data)
        predict = rf.predict(data)
        output = ""
        if predict == 0:
            output = "High"
        elif predict == 1:
            output = "Low"
        elif predict == 2:
            output = "Medium"
        today = str(date.today())    
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'lungdisease',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO patients(patient_name,patient_data,predicted_stage,process_date) VALUES('"+uname+"','"+input_values+"','"+output+"','"+today+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        context= {'data':"Cancer Stage Predicted As : "+output}
        return render(request, 'PatientScreen.html', context)
            

def TrainML(request):
    if request.method == 'GET':
        global p, r, f, a
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th>'
        output += '<font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        output+='</tr>'
        algorithms = ['Random Forest']
        output+='<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(a)+'</td>'
        output+='<td><font size="" color="black">'+str(p)+'</td><td><font size="" color="black">'+str(r)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
        output+= "</table></br></br></br>"
        context= {'data':output}
        return render(request, 'PatientScreen.html', context)        

def PatientLogin(request):
    if request.method == 'GET':
       return render(request, 'PatientLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def DoctorLogin(request):
    if request.method == 'GET':
       return render(request, 'DoctorLogin.html', {})

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        utype = request.POST.get('t6', False)
        
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'lungdisease',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'lungdisease',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact_no,email,address,usertype) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"','"+utype+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup completed<br/>You can login with "+username
        context= {'data': status}
        return render(request, 'Register.html', context)

def sendOTP(email, otp_value):
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'kaleem202120@gmail.com'
        email_password = 'xyljzncebdxcubjq'
        connection.login(email_address, email_password)
        connection.sendmail(from_addr="kaleem202120@gmail.com", to_addrs=em, msg="Subject : Your OTP for login: "+otp_value)    

def OTPAction(request):
    if request.method == 'POST':
        global uname, utype, otp
        otp_value = request.POST.get('t1', False)
        if otp == otp_value:
            context= {'data':'Welcome '+uname}
            return render(request, 'PatientScreen.html', context)
        else:
            context= {'data':'Invalid OTP! Please retry'}
            return render(request, 'OTP.html', context)

def PatientLoginAction(request):
    if request.method == 'POST':
        global uname, utype, otp
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'lungdisease',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password, usertype, email FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1] and row[2] == 'Patient':
                    email = row[3]
                    uname = username
                    utype = "Patient"
                    otp = str(random.randint(1000, 9999))
                    index = 1
                    sendOTP(email, otp)
                    break		
        if index == 1:
            context= {'data':'OTP sent to your mail to continue login'}
            return render(request, 'OTP.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'PatientLogin.html', context)                 


def DoctorLoginAction(request):
    if request.method == 'POST':
        global uname, utype, otp
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'lungdisease',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password, usertype, email FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1] and row[2] == 'Doctor':
                    email = row[3]
                    uname = username
                    utype = "Doctor"
                    index = 1
                    break		
        if index == 1:
            context= {'data':'Welcome '+uname}
            return render(request, 'DoctorScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'DoctorLogin.html', context)                 

        
