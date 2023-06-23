#####################################################
from flask import Flask, render_template, request
from numpy import *
from sklearn import linear_model
from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
from sklearn.datasets import load_files
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import sqlite3

df= pd.read_csv('data.csv',encoding='ISO-8859-1')
df = df[0:2500]
df = df[['Type','Posts']]
df = df.dropna()
#Cleaning data
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['Type']= label_encoder.fit_transform(df['Type'])

Tweet = []
Labels = []

for row in df["Posts"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)



df['message']=df['Posts']


#df = df[0:2000]
X = df['message']
y = df['Type']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
estimator = []
estimator.append(('MLp', 
                  MLPClassifier(hidden_layer_sizes=(250,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)))
estimator.append(('RFC', RandomForestClassifier()))
estimator.append(('DTC', DecisionTreeClassifier()))
vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
vot_hard.fit(X_train, y_train)
predictions = vot_hard.predict(X_test)

from sklearn.metrics import accuracy_score
val8 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for Voting Classifier: ", val8, "\n")


#####################################################

app = Flask(__name__)

#####################################################
# Loading Dataset Globally
data = pd.read_csv("dataset.csv")
array = data.values

for i in range(len(array)):
    if array[i][0] == "Male":
        array[i][0] = 1
    else:
        array[i][0] = 0

df = pd.DataFrame(array)

maindf = df[[0, 1, 2, 3, 4, 5, 6]]
mainarray = maindf.values

temp = df[7]
train_y = temp.values
train_y = temp.values

for i in range(len(train_y)):
    train_y[i] = str(train_y[i])

mul_lr = linear_model.LogisticRegression(
    multi_class="multinomial", solver="newton-cg", max_iter=1000
)
mul_lr.fit(mainarray, train_y)
#####################################################

@app.route('/predict1',methods=['POST'])
def predict1():

    if request.method == 'POST':
        
        message = request.form['message']
        data = [message]
        #cv = tfidf.fit_transform(
        vect = cv.transform(data).toarray()
        my_prediction = vot_hard.predict(vect)
        
        print(my_prediction[0])
        return render_template('result1.html',prediction = my_prediction[0],message=message)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    else:
        age = int(request.form["age"])
        if age < 17:
            age = 17
        elif age > 28:
            age = 28

        inputdata = [
            [
                request.form["gender"],
                age,
                9 - int(request.form["openness"]),
                9 - int(request.form["neuroticism"]),
                9 - int(request.form["conscientiousness"]),
                9 - int(request.form["agreeableness"]),
                9 - int(request.form["extraversion"]),
            ]
        ]

        for i in range(len(inputdata)):
            if inputdata[i][0] == "Male":
                inputdata[i][0] = 1
            else:
                inputdata[i][0] = 0

        df1 = pd.DataFrame(inputdata)
        testdf = df1[[0, 1, 2, 3, 4, 5, 6]]
        maintestarray = testdf.values

        y_pred = mul_lr.predict(maintestarray)
        for i in range(len(y_pred)):
            y_pred[i] = str((y_pred[i]))
        DF = pd.DataFrame(y_pred, columns=["Predicted Personality"])
        DF.index = DF.index + 1
        DF.index.names = ["Person No"]

        return render_template(
            "result.html", per=DF["Predicted Personality"].tolist()[0]
        )

@app.route('/')
def home():
	return render_template('home.html')

@app.route("/signup")
def signup():
    
    
    name = request.args.get('username','')
    number = request.args.get('number','')
    email = request.args.get('email','')
    password = request.args.get('psw','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `detail` (`name`,`number`,`email`, `password`) VALUES (?, ?, ?, ?)",(name,number,email,password))
    con.commit()
    con.close()

    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('name','')
    password1 = request.args.get('psw','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `name`, `password` from detail where `name` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()
    print(data)

    if data == None:
        return render_template("signup.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/working")
def working():
    return render_template("working.html")


@app.route('/logon')
def reg():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')

# Handling error 404
@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", code=404, text="Page Not Found"), 404


# Handling error 500
@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", code=500, text="Internal Server Error"), 500

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/index1')
def index1():
	return render_template('index1.html')

if __name__ == "__main__":

    # use 0.0.0.0 for replit hosting
    app.run(debug=True)

    # for localhost testing
    # app.run()
