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
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import preprocessing
import tkinter as tk
from tkinter import ttk
from collections import Counter
from PIL import ImageTk, Image
import math
def menu():
    #for testing stuff in command line
    menuinput = input("do you want to: \n a: create a wordcloud \n b: view information about keywords \n c: test the model for yourself \n d: run the model again \n e: run the gui")
    if menuinput ==  "a":
        wordcloud(finaldf)
    elif menuinput == "b":
        keywords(finaldf)
    elif menuinput == "c":
        inputtester(vectorizer, lr)
    elif menuinput == "d":
        model()
    elif menuinput == "e":
        gui()
    else:
        print("invalid input")
        menu()

def remove_punctuation(text):
    #subroutine to remove punctuation from stuff
    finaltext="".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return finaltext


def dataframes():
    #sub for creating a few vital dataframes & series
    #other series and such are created as and when needed
    #brings in all the data from the csv file
    global fulldf
    fulldf=pd.read_csv("posts.csv")  
    fulldf.head()

    #removes punctuation from relevant fields
    fulldf["Post_Text"] = fulldf["Post_Text"].apply(remove_punctuation)
    #split into 2 dataframes, 1 left wing, 1 right wing
    global leftdf
    global rightdf
    leftdf = fulldf[fulldf["orientation"] == 0]
    rightdf = fulldf[fulldf["orientation"] == 1]
    global dfseries
    dfseries = pd.Series(fulldf['Post_Text'].values)
    #creating a final dataframe with only relevant info for training/testing
    global finaldf
    finaldf = fulldf[["Post_Text", "orientation"]]
    finaldf.head()

#subroutine that actually does the training and testing
def model():
    #doing the train test split
    index = finaldf.index
    finaldf["random"] = np.random.randn(len(index))
    train = finaldf[finaldf["random"] <= 0.8]
    test = finaldf[finaldf['random'] > 0.8]

    #count vectorizer
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
    #y_pred = log.predict(X_test)
    global score
    score = (round((accuracy_score(predictions,y_test))*100, 1), "%")
    global classreport
    classreport = (classification_report(predictions,y_test))
    print(classreport)
    print(score)
    print("model complete, running GUI")
    gui()
    #menu()
#allows users to input their own text to be predicted
def inputtester(vectorizer, lr):
    #takes user input and puts it into correct format
    userinput = input("try the model for yourself, input some text. Please note that the model has been trained on longer (200+ character) strings: ")
    userinput = remove_punctuation(userinput)
    userinput = [userinput]
    cleaninput = vectorizer.transform(userinput)
    #runs the model and outputs conclusion
    inputpredict = lr.predict(cleaninput)
    if inputpredict==[1]:
        print("I predict this is a right leaning comment")
    elif inputpredict==[0]:
        print("I predict this is a left leaning comment")
    else:
        print("something has gone very wrong for this to display")
    rerun()
    menu()
def rerun():
    #allows the user to enter more custom text
    rerun = input("enter some more text? (y/n)")
    if rerun == "y":
        inputtester(vectorizer, lr)
    elif rerun == "n":
        print("program end")
    else:
        print("input not recognised")
        rerun()

        
def keywords(dataframe):
    #allows users to see the number of occurances of keywords in the dataset.
    #updating now to do other stuff

    #prompts the user for an input then converts it to lower case
    keyword = input("input a keyword: ")
    keyword = keyword.lower()
    
    #create a subseries of all the rows of the data that contain the keyword
    #dfseries = pd.Series(dataframe['Post_Text'].values)
    subseries = dfseries[dfseries.str.contains(keyword)]
    print(subseries) #debugging line remove when complete
    leftcounter = 0
    rightcounter = 0
    #initalises a seperate series of all the orientation values and the user counter
    orientationseries = pd.Series(dataframe['orientation'].values)
    userseries = pd.Series(fulldf['User_Hash'].astype('str').values)
    bag_words = Counter()
    
    #checks the orientations of every post that contains the keyword
    #works by comparing the keyword subseries with the orientation series
    #also keeps track of how many different users have written posts using the keyword
    for i, j in subseries.iteritems():
        if orientationseries.loc[i]==0:
            leftcounter +=1
        elif orientationseries.loc[i]==1:
            rightcounter +=1
        userupdate = {userseries.loc[i] : 1}
        bag_words.update(userupdate)
        

    #counts the number of posts the keyword features in and the number of unique users who posted using that keyword
    counter = len(subseries)
    usercount = len(bag_words)
    datasetlen = len(dfseries)
    #counts the number of times the keyword appears
    totalcounter = subseries.str.count(keyword).sum()

    #does some output
    print("the word ", keyword, " appears", totalcounter, "times in ", counter, " posts out of a total of", datasetlen, "posts")
    rightpercent = round(((rightcounter/counter) * 100), 2)
    leftpercent = round(((leftcounter/counter) * 100), 2)
    print("this is made up of ", rightpercent, "% in right leaning subreddits, and ", leftpercent, "% in left leaning ones")
    print("posts came from ", usercount, " users, with 1 user posting ", (bag_words.most_common(1)[0][1]), "seperate times")
    decision = input("view all the posts and replies from the top user? y/n")

    #meant to display posts from top user, doesn't work yet
    if decision == 'y':
        topuserseries = userseries[userseries.str.contains(bag_words.most_common(1)[0][0])]
        count1=0
        for i, j in topuserseries.iteritems():
            print("-----post starts-----")
            print(subseries.iloc[i])
            print("-----post ends-------")
            count1 = count1 + 1
        print(count1)
            
    keywords(finaldf) #runs again for debugging purposes            
    menu()
    
def wordcloud(series):
    #creates wordcloud of titles from the full data
    #can probably do some other things
    #just using it to test atm
    #dfseries = pd.Series(dataframe['Post_Text'].values)
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    text = " ".join(review for review in series.values.astype("U"))
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordcloud.png")
    wordcloudGUI()
    
def wordcloudGUI():
    #displays wordcloud in the GUI
    cloudGUI = tk.Toplevel(root)
    cloudGUI.title("Wordcloud")
    wcimage = Image.open("wordcloud.png")
    test = ImageTk.PhotoImage(wcimage)
    label1 = ttk.Label(cloudGUI, image=test)
    label1.image = test
    label1.pack()
    cloudGUI.mainloop()
def modelinformationGUI():
    infoPanel = tk.Toplevel(root)
    infoPanel.title("Model Information")
    datalen = len(dfseries)
    lbltext = "the model uses logistic regression to predict the political leaning of a given reddit post \nFor this running, the dataset had ", datalen ," posts in it."
    accuracytext = "On this occaision, the model was ", score , " accurate in its own tests."
    infolbl = ttk.Label(infoPanel, text = lbltext)
    infolbl2 = ttk.Label(infoPanel, text = accuracytext)
    infolbl.pack()
    infolbl2.pack()
    infoPanel.mainloop()
def gui():
    #builds and controls the GUI homepage
    global root
    root = tk.Tk()
    root.title("Main Window")
    #main window buttons
    modelinfobutton = ttk.Button(root, text='view model information', command=modelinformationGUI)
    modelrunbutton = ttk.Button(root, text='run model again', command=model)
    keywordsbutton = ttk.Button(root, text='view keywords information', command=lambda: keywords(finaldf))
    inputtesterbutton = ttk.Button(root, text='test the model for yourself', command=lambda: inputtester(vectorizer, lr))
    inputtesterbutton.pack()
    keywordsbutton.pack()
    modelrunbutton.pack()
    modelinfobutton.pack()
    newwindowbtn = ttk.Button(root, text ="Create Wordclouds", command = cloudDesignGUI)
    newwindowbtn.pack()
    
    root.mainloop()


def cloudDesignGUI():
    #opens new GUI window for the creation of different wordclouds
    designWindow = tk.Toplevel(root)
    designWindow.title("Wordcloud Design")
    designWindowlbl = ttk.Label(designWindow, text ="Click a button to view a pre-designed wordcloud, or create your own") 
    selectBtn1 = ttk.Button(designWindow, text='Use Full Dataset', command=lambda: wordcloud(pd.Series(finaldf['Post_Text'].values)))
    selectBtn2 = ttk.Button(designWindow, text='Use only left leaning subreddits', command=lambda: wordcloud(pd.Series(leftdf['Post_Text'].values)))
    selectBtn3 = ttk.Button(designWindow, text='Use only right leaning subreddits', command=lambda: wordcloud(pd.Series(rightdf['Post_Text'].values)))
    designWindowlbl.pack()
    selectBtn1.pack()
    selectBtn2.pack()
    selectBtn3.pack()
    designWindow.mainloop()
dataframes()
model()
#keywords(finaldf)
#wordcloud(finaldf)
#inputtester(vectorizer, lr)
#gui()



    
    
