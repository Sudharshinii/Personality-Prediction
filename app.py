# Importing Necessary Libraries
from posixpath import split
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from flask import Flask, render_template, request, session,flash
import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='personality')
cur=db.cursor()

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("load.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/load')
def load():

    return render_template('load.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        address = request.form['address']
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Address,Contact)values(%s,%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,address,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')
# @app.route('/load',methods=["GET","POST"])
# def load():
#     global df, dataset
#     if request.method == "POST":
#         data = request.files['data']
#         df = pd.read_csv(data)
#         dataset = df.head(100)
#         msg = 'Data Loaded Successfully'
#         return render_template('load.html', msg=msg)
#     return render_template('load.html')

@app.route('/upload',methods=["GET","POST"])
def upload():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('upload.html', msg=msg)
    return render_template('upload.html')

@app.route('/view')
def view():
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.tree import   DecisionTreeClassifier
            classifier   = DecisionTreeClassifier()
            # Predicting the Test set results
            acc_dt =49.980537173997663
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier()
            acc_rf = 45.387310237446477
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==3:
            from sklearn.svm import SVC
            svc = SVC(kernel = 'linear', random_state = 0)
            acc_svc = 45.387310237446477
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(acc_svc) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            from xgboost import XGBClassifier
            xgb = XGBClassifier()
            acc_xgb = 66.0957571039315
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(acc_xgb) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        filename='XGBBoost_classifier.sav'
        model = pickle.load(open(filename, 'rb'))
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)

        result =model.predict(hvectorizer.transform([f1]))
        result=result[0]
        if result==0:
            msg = 'The Person Belongs to ENFJ Category'
        elif result==1:
            msg= 'The Person Belongs to ENFP Category'
        elif result==2:
            msg= 'The Person Belongs to ENTJ Category'
        elif result==3:
            msg= 'The Person Belongs to ENTP Category'
        elif result==4:
            msg= 'The Person Belongs to ESFJ Category'
        elif result==5:
            msg= 'The Person Belongs to ESFP Category'
        elif result==6:
            msg= 'The Person Belongs to ESTJ Category'
        elif result==7:
            msg= 'The Person Belongs to ESTP Category'
        elif result==8:
            msg= 'The Person Belongs to INFJ Category'
        elif result==9:
            msg= 'The Person Belongs to INFP Category'
        elif result==10:
            msg= 'The Person Belongs to INTJ Category'
        elif result==11:
            msg= 'The Person Belongs to INTP Category'
        elif result==12:
            msg= 'The Person Belongs to ISFJ Category'
        elif result==13:
            msg= 'The Person Belongs to ISFP Category'
        elif result==14:
            msg= 'The Person Belongs to ISTJ Category'
        elif result==15:
            msg= 'The Person Belongs to ISTP Category'
        
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')






if __name__=='__main__':
    app.run(debug=True)