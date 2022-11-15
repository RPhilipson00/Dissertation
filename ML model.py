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
fulldf=pd.read_csv("posts.csv") #brings in all the data from the csv file 
fulldf.head()

#split into 2 dataframes, 1 left wing, 1 right wing.
leftdf = fulldf[fulldf["orientation"] == 0]
rightdf = fulldf[fulldf["orientation"] == 1]


def remove_punctuation(text):
    #subroutine to remove punctuation from stuff
    finaltext="".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return finaltext

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


# count vectorizer:
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Post_Text'].values.astype("U"))
test_matrix = vectorizer.transform(test['Post_Text'].values.astype("U"))


# Logistic Regression
lr = LogisticRegression(max_iter=5000)
X_train = train_matrix
X_test = test_matrix
y_train = train['orientation']
y_test = test['orientation']
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)


# find accuracy, precision, recall:

new = np.asarray(y_test)
confusion_matrix(predictions,y_test)
print(classification_report(predictions,y_test))

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

#wordcloud()
