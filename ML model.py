#dissertation project ML model
#Robert Philipson


#import absolutely heaps of stuff
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


fulldf=pd.read_csv("posts.csv") #brings in all the data from the csv file 
fulldf.head()

#split into 2 dataframes, 1 left wing, 1 right wing.
leftdf = fulldf[fulldf["orientation"] == 0]
rightdf = fulldf[fulldf["orientation"] == 1]


def remove_punctuation(text):
    #subroutine to remove punctuation from stuff
    finaltext="".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return finaltext

fulldf["Title"] = fulldf["Title"].apply(remove_punctuation)


finaldf = fulldf[["Title", "orientation"]]
finaldf.head()

index = finaldf.index
finaldf["random"] = np.random.randn(len(index))

train = finaldf[finaldf["random"] <= 0.8]
test = finaldf[finaldf['random'] > 0.8]

def wordcloud():
    #creates wordcloud of titles
    #can probably do some other things
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    textt = " ".join(review for review in rightdf.Title)
    wordcloud = WordCloud(stopwords=stopwords).generate(textt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud11.png')
    plt.show()

wordcloud()
