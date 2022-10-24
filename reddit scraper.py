#dissertation project subreddit reader

import praw
import pandas as pd
redditRead = praw.Reddit(client_id="ZHd4uF6mhAs2e8_rFnJLnA",
                                client_secret="OlnDy_ZTJLX9CHXXGnJ4-3Z-fRbFOg",
                                user_agent="Dissertation Scraper",
                                username="Dissbot2000",
                                password="MEGAsecretdisspass2022")
#Creates an authorised praw instance to read reddit
posts_dict1 = {"Post ID": [],"Title": [], "Post Text": []}
#initialises pandas dataframe


def scraper(sub):
        posts = redditRead.subreddit(sub)
        #redditor = reddit.redditor(redditor_name)
        #reads the subreddit
        print("reading sub:", sub) #debugging line
        for post in posts.hot(limit=1000):  #update to bigger number 
                posts_dict1["Title"].append(post.title)
                posts_dict1["Post Text"].append(post.selftext)
                posts_dict1["Post ID"].append(post.id)
                
                #posts_dict1["User"].append(post.author)
        #takes specific fields from the posts, builds list


# Saving the data in a pandas dataframe and exporting to csv

subreddit_list = ["communism","socialism", "republican", "libertarian","labouruk","tories","greenandpleasant","conservative","anarcho_capitalism","trump"]
for sub in subreddit_list:
        scraper(sub)
        #loops through list of subreddits i've chosen, and runs the scraper.
posts_file = pd.DataFrame(posts_dict1)
posts_file.to_csv("posts.csv", mode="a", index=True, header=False)
#appends the 

