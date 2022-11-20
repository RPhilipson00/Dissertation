#dissertation project ML model
#Robert Philipson


#import absolutely heaps of stuff in no particular order
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
import tkinter as tk


def remove_punctuation(text):
    #subroutine to remove punctuation from stuff
    finaltext="".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return finaltext

#subroutine that actually does the training and testing
def model():
    #brings in all the data from the csv file
    fulldf=pd.read_csv("posts.csv")  
    fulldf.head()

    #split into 2 dataframes, 1 left wing, 1 right wing
    #not actually needed for the model but probably useful for other stuff
    leftdf = fulldf[fulldf["orientation"] == 0]
    rightdf = fulldf[fulldf["orientation"] == 1]

    #removes punctuation from relevant fields
    fulldf["Post_Text"] = fulldf["Post_Text"].apply(remove_punctuation)

    #creating a final dataframe with only relevant info
    finaldf = fulldf[["Post_Text", "orientation"]]
    finaldf.head()


    #doing the train test split
    index = finaldf.index
    finaldf["random"] = np.random.randn(len(index))
    train = finaldf[finaldf["random"] <= 0.8]
    test = finaldf[finaldf['random'] > 0.8]

    #count vectorizer, 
    global vectorizer
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['Post_Text'].values.astype("U"))
    test_matrix = vectorizer.transform(test['Post_Text'].values.astype("U"))

    #run the lr model
    global lr
    lr = LogisticRegression(solver="lbfgs", max_iter=5000)
    X_train = train_matrix
    X_test = test_matrix
    y_train = train['orientation']
    y_test = test['orientation']
    lr.fit(X_train,y_train)
    predictions = lr.predict(X_test)

    #determine accuracy and print data
    new = np.asarray(y_test)
    confusion_matrix(predictions,y_test)
    print(classification_report(predictions,y_test))

#allows users to input their own text to be predicted
def inputtester(vectorizer, lr):
    userinput = input("try the model for yourself, input some text: ")
    userinput = remove_punctuation(userinput)
    userinput = [userinput]
    cleaninput = vectorizer.transform(userinput)
    inputpredict = lr.predict(cleaninput)
    if inputpredict==[1]:
        print("I predict this is a right wing comment")
    elif inputpredict==[0]:
        print("I predict this is a left wing comment")
    else:
        print("something has gone very wrong for this to display")
    rerun()
    
def rerun():
    rerun = input("enter some more text? (y/n)")
    if rerun == "y":
        inputtester(vectorizer, lr)
    elif rerun == "n":
        print("program end")
    else:
        print("input not recognised")
        rerun()
        
def wordcloud():
    #creates wordcloud of titles from the full data
    #can probably do some other things
    #just using it to test atm
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    textt = " ".join(review for review in leftdf.Post_Text.values.astype("U"))
    wordcloud = WordCloud(stopwords=stopwords).generate(textt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloudrw.png')
    plt.show()

def gui():
    window = tk.Tk()
    #label = tk.Label(text="Test")
    #label.pack()
    button = tk.Button(text="click me", width=25, height=5, bg="blue", fg="yellow")
    button.pack()
    window.mainloop()
#wordcloud()
model()
inputtester(vectorizer, lr)
#gui()
