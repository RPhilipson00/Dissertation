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
import csv
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
from tkinter import ttk, filedialog
from tkinter import *
from collections import Counter
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import random
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
    global userHashes
    userHashes = (fulldf['User_Hash'].unique())
    global subs
    subs = pd.Series((fulldf['sub'].unique()).astype('str'))
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
    score = (round((accuracy_score(predictions,y_test))*100, 1))
    global classreport
    classreport = (classification_report(predictions,y_test))
    print(classreport)
    print(score)
    print("model complete, running GUI")
    gui()
    #menu()
#allows users to input their own text to be predicted
def inputtester(cleaninput):
    #runs the model and outputs conclusion
    inputpredict = lr.predict(cleaninput)
    if inputpredict==[1]:
        prediction = "I predict this is a right leaning comment"
    elif inputpredict==[0]:
        prediction = "I predict this is a left leaning comment"
    return prediction
    #rerun()
    #menu()
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

def lengthfinder(df, sub):
    #finds the number of posts from a given subreddit in the dataset
    subdf = df[df['sub'] == sub]
    return len(subdf.index)

def keywords(dataframe, keyword):
    #allows users to see the number of occurances of keywords in the dataset.
    #updating now to do other stuff

    #prompts the user for an input then converts it to lower case, since dataset is lower case
    keyword = keyword.lower()
    
    #create a subseries of all the rows of the data that contain the keyword
    subseries = dfseries[dfseries.str.contains(keyword)]
    print(subseries) #debugging line remove when complete
    leftcounter = 0
    rightcounter = 0
    #initalises a seperate series of all the orientation values and the user counter
    orientationseries = pd.Series(dataframe['orientation'].values)
    subnameseries = pd.Series(dataframe['sub'].astype('str').values)
    userseries = pd.Series(dataframe['User_Hash'].astype('str').values)
    bag_words = Counter()
    counters = [["republican", 0], ["libertarian", 0] ,["tories",0] ,["conservative", 0],["anarcho_capitalism", 0],["trump", 0], ["louderwithcrowder", 0], ["communism", 0],["socialism", 0],["labouruk", 0],["greenandpleasant",0],["democrats",0], ["anarchocommunism",0],["anarchism", 0]]
    
    #checks the orientations of every post that contains the keyword
    #works by comparing the keyword subseries with the orientation series
    #also keeps track of how many different users have written posts using the keyword
    for i, j in subseries.iteritems():
        if orientationseries.loc[i]==0:
            leftcounter +=1
        elif orientationseries.loc[i]==1:
            rightcounter +=1
        for x in range (len(counters)):
            if subnameseries.loc[i]== counters[x][0]:
                counters[x][1] +=1
        userupdate = {userseries.loc[i] : 1}
        bag_words.update(userupdate)
        #subname = subnameseries.loc[i]
       

        #subcounter(subname)  
    #counts the number of posts the keyword features in and the number of unique users who posted using that keyword
    counter = len(subseries)
    usercount = len(bag_words)
    datasetlen = len(dfseries)
    #counts the number of times the keyword appears
    totalcounter = subseries.str.count(keyword).sum()

    if totalcounter != 0:
        #if the keyword appears in the dataset then some interesting stats are printed
        rightpercent = round(((rightcounter/counter) * 100), 2)
        leftpercent = round(((leftcounter/counter) * 100), 2)
        leftsize = len(leftdf.index)
        rightsize = len(rightdf.index)
        leftproportion = round(((leftcounter/leftsize) * 100),2)
        rightproportion = round(((rightcounter/rightsize) * 100),2)
        linex = ""
        for x in range (len(counters)):
            #works out number of times keyword appears in each subreddit
            xlines = "this word appeared ", counters[x][1], " times in the subreddit r/", counters[x][0], " out of ", lengthfinder(fulldf, counters[x][0]), " posts \n"
            st = ''.join(map(str, xlines))
            linex = linex + st
        output = ("the word ", keyword, " appears ", str(totalcounter), " times in ", str(counter), " posts out of a total of ", str(datasetlen), " posts \n",
                  "this is made up of ", str(rightpercent), "% in right leaning subreddits, and ", str(leftpercent), "% in left leaning ones \n",
                  "it appeared in ", str(leftproportion), "% of posts in left leaning subreddits \n",
                  "and in ", str(rightproportion), "% of posts in right leaning subreddits \n",
                  str(linex), "posts came from ", str(usercount), " users, with 1 user posting ", str((bag_words.most_common(1)[0][1])), " seperate times",)
        strout = ''.join(map(str, output))
        proportions = []
        labels = []
        countvals=[]
        for x in range (len(counters)):
            #works out the proportion of each subreddit
            proportion = round(((counters[x][1]/counter)*100), 2)
            if proportion != 0:
                proportions.append(proportion)
                labels.append(counters[x][0])
                countvals.append(counters[x][1])
                
        #displays the info in the GUI
        keyInfoWindow = tk.Toplevel(root)
        keyInfoWindow.title("Keyword Info")
        mainInfoLbl = ttk.Label(keyInfoWindow, text=strout)
        pieBtn = ttk.Button(keyInfoWindow, text="create a pie chart", command = lambda: piechart(proportions,labels))
        barBtn = ttk.Button(keyInfoWindow, text="create a bar chart", command = lambda: bar(labels,countvals))
        mainInfoLbl.pack()
        pieBtn.pack()
        barBtn.pack()
        #makes a pie chart and a bar chart
        def piechart(proportions, labels):
            piechartfig = plt.pie(proportions, labels = labels, autopct='%.2f')
            plt.savefig("pie.png")

        def bar(labels, countvals):
            barchartfig = plt.barh(labels,countvals)
            plt.savefig("bar.png")
        
        
        #figure = plt.Figure(figsize=(2,2), dpi=100)
        #pie = FigureCanvasTkAgg(figure, keyInfoWindow)
        #pie.get_tk_widget().pack()
        #ax1=figure.add_subplot(221)
        #ax1.plot(plt.pie(proportions, labels = labels, autopct='%.2f'))
        #plt.show()
        #mainInfoLbl.pack()
    else:
      print("this word does not appear in the dataset")



            
    #keywords(finaldf) #runs again for debugging purposes            
    #menu()
    
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
    print(Counter(text.split()).most_common(10))
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
    lbltext = "the model uses logistic regression to predict the political leaning of a given reddit post\nFor this running, the dataset had "+ str(datalen) +" posts and comments in it"
    accuracytext = "On this occaision, the model was "+ str(score) + "% accurate in its own tests."
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
    keywordsbutton = ttk.Button(root, text='view keywords information', command=lambda: keywordsGUI())
    inputtesterbutton = ttk.Button(root, text='test the model for yourself', command=lambda: modelTestGUI())
    userinfobutton = ttk.Button(root, text='view user information', command=randomUserGUI)
    inputtesterbutton.pack()
    keywordsbutton.pack()
    modelrunbutton.pack()
    modelinfobutton.pack()
    newwindowbtn = ttk.Button(root, text ="Create Wordclouds", command = cloudDesignGUI)
    newwindowbtn.pack()
    userinfobutton.pack()
    root.mainloop()

def keywordsGUI():
    keywordsWindow = tk.Toplevel(root)
    keywordsWindow.title("Keyword explorer")
    keywordsWindowlbl = ttk.Label(keywordsWindow, text="enter a word to see how it appears in the data")
    inputbox = tk.Entry(keywordsWindow)
    enterbtn = ttk.Button(keywordsWindow, text="Enter", command=lambda: keywords(fulldf, inputbox.get()))
    keywordsWindowlbl.pack()
    inputbox.pack()
    enterbtn.pack()

def modelTestGUI():
    modelTestWindow = tk.Toplevel(root)
    modelTestWindow.title("Model Tester")
    modelTestWindowlbl = ttk.Label(modelTestWindow, text="Enter a sentence for the model to predict")
    outputlbl = ttk.Label(modelTestWindow, text=" ")
    inputbox = tk.Entry(modelTestWindow)
    userinput = inputbox.get()
    userinput = remove_punctuation(userinput)
    userinput = [userinput]
    cleaninput = vectorizer.transform(userinput)
    label_file_explorer = ttk.Label(modelTestWindow, text="Or pick a csv file containing posts")
    filebtn = ttk.Button(modelTestWindow, text = "file explorer", command =lambda: outputlbl.config(text = (browseFiles())))
    #outputs prediction to gui
    enterbtn = ttk.Button(modelTestWindow, text="Enter", command=lambda: outputlbl.config(text= (inputtester(cleaninput))))
    modelTestWindowlbl.pack()
    inputbox.pack()
    outputlbl.pack()
    enterbtn.pack()
    label_file_explorer.pack()
    filebtn.pack()

##def keywordInfoGUI(strout):
##    keyInfoWindow = tk.Toplevel(root)
##    keyInfoWindow.title("Keyword Info")
##    mainInfoLbl = ttk.Label(keyInfoWindow, text=strout)
    
   # mainInfoLbl.pack()
def cloudDesignGUI():
    #opens new GUI window for the creation of different wordclouds
    designWindow = tk.Toplevel(root)
    designWindow.title("Wordclouds")
    designWindowlbl = ttk.Label(designWindow, text ="Click a button to view a pre-designed wordcloud, or create your own") 
    selectBtn1 = ttk.Button(designWindow, text='Use Full Dataset', command=lambda: wordcloud(pd.Series(finaldf['Post_Text'].values)))
    selectBtn2 = ttk.Button(designWindow, text='Use only left leaning subreddits', command=lambda: wordcloud(pd.Series(leftdf['Post_Text'].values)))
    selectBtn3 = ttk.Button(designWindow, text='Use only right leaning subreddits', command=lambda: wordcloud(pd.Series(rightdf['Post_Text'].values)))
    dropDownOptions = subs
    clicked = tk.StringVar()
    clicked.set(subs[0])
    
    drop = OptionMenu(designWindow, clicked, *dropDownOptions)
    def subcloud(menuInput):
        subname = menuInput
        subdf = fulldf[fulldf['sub'] == subname]
        relatedSeries = pd.Series(subdf['Post_Text'].values)
        wordcloud(relatedSeries)
    generateBtn = ttk.Button(designWindow, text = 'use a single sub', command = lambda: subcloud(clicked.get()))
    designWindowlbl.pack()
    selectBtn1.pack()
    selectBtn2.pack()
    selectBtn3.pack()
    drop.pack()
    generateBtn.pack()
    designWindow.mainloop()

def randomUserGUI():
    #user information
    print(len(userHashes), "unique users")
    userWindow = tk.Toplevel(root)
    userWindow.title("User Information")
    userWindowlbl1 = ttk.Label(userWindow, text ="there are " + str(len(userHashes)) + " unique users in the dataset")
    twoSubCount = 0
    fiveSubCount = 0
    for x in range(len(userHashes)):
        userdf = fulldf[fulldf['User_Hash'] == userHashes[x]]
        subseries=pd.Series(userdf['sub'].values)
        unique = subseries.unique()
        if len(unique)>1:
            twoSubCount+=1
        if len(unique)>4:
            fiveSubCount+=1
    dropDownOptions = ['1','2','3','4','5']
    clicked = tk.StringVar()
    clicked.set('1')
    drop = OptionMenu(userWindow, clicked, *dropDownOptions)
    
    
    userWindowlbl2 = ttk.Label(userWindow, text ="Select an option to view data about a random user who posted in more than one subreddit")
    userWindowlbl3 = ttk.Label(userWindow, text = "of them, "+ str(twoSubCount)+ " users posted in more than one subreddit")
    userWindowlbl4 = ttk.Label(userWindow, text = str(fiveSubCount) + " users posted in 5 different subreddits")
    generateBtn = ttk.Button(userWindow, text="pick a random user", command = lambda:  userInfo(int(clicked.get())))
    userWindowlbl1.pack()
    userWindowlbl3.pack()
    userWindowlbl4.pack()
    userWindowlbl2.pack()
    drop.pack()
    generateBtn.pack()
def userInfo(number):
    #creates relationship diagram
    user = random.choice(userHashes)
    userdf = fulldf[fulldf['User_Hash'] == user]
    userSubredditSeries = pd.Series(userdf['sub'].values)
    uniquesubs = (userSubredditSeries.unique())
    
    if len(uniquesubs) >=number:
        userPostSeries = pd.Series(userdf['Post_Text'].values)
        userOrientationSeries = pd.Series(userdf['orientation'].values)
        userGraph = nx.Graph()
        userGraph.add_node(user)
        userGraph.add_nodes_from(uniquesubs)

        for x in range(len(uniquesubs)):
            subdf = userdf[userdf['sub']==uniquesubs[x]]
            length = subdf.shape[0]
            userGraph.add_edge(user,uniquesubs[x],weight=length)
        print("edge set: ", userGraph.edges())
        pos=nx.spring_layout(userGraph, seed=7)
        nx.draw_networkx_nodes(userGraph,pos)
        nx.draw_networkx_edges(userGraph, pos,edgelist=userGraph.edges, width=6, alpha=0.5, edge_color="b", style="dashed")
        nx.draw_networkx_labels(userGraph,pos)
        edge_weight = nx.get_edge_attributes(userGraph,'weight')
        
        nx.draw_networkx_edge_labels(userGraph, pos, edge_labels = edge_weight)
        ax=plt.gca()
        ax.margins(0.08)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
                                    
##        leftcount = 0
##        rightcount = 0
##        bothtrue = False
##        for x in range(0,20):
##            for i in userOrientationSeries:
##                if i == 0:
##                    leftcount+=1
##                elif i == 1:
##                    rightcount+=1
##
##            if leftcount>0 and rightcount>0:
##                bothtrue = True
##            x+=1
##        print("bothtrue: ", bothtrue)
    else:
        print("no")
        userInfo(number)

def browseFiles():
    #opens a file explorer, allowing the user to get the model to work on their own dataset
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
      
    # Change label contents
    userdf=pd.read_csv(filename)  
    userdf.head()

    #removes punctuation from relevant fields
    userdf["Post_Text"] = userdf["Post_Text"].apply(remove_punctuation)
    posts = vectorizer.transform(userdf['Post_Text'].values.astype("U"))
    predictions = lr.predict(posts)
    userdf["predictions"] = predictions
    userdf.to_csv("custom predictions.csv", mode="a", index=False, header=True)
    leftcount = 0
    rightcount = 0
    for x in range (len(predictions)):
        if predictions[x]==0:
            leftcount+=1
        else:
            rightcount+=1
    return("dataset contains " , str(leftcount) , " left wing and " , str(rightcount) , " right wing")
    
dataframes()
model()

                            
#keywords(finaldf)
#wordcloud(finaldf)
#inputtester(vectorizer, lr)
#gui()



    
    
