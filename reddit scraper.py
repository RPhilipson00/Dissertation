#dissertation project subreddit reader
#Robert Philipson


import praw
import pandas as pd
import nltk
redditRead = praw.Reddit(client_id="ZHd4uF6mhAs2e8_rFnJLnA",
                                client_secret="OlnDy_ZTJLX9CHXXGnJ4-3Z-fRbFOg",
                                user_agent="Dissertation Scraper",
                                username="Dissbot2000",
                                password="MEGAsecretdisspass2022")
#Creates an authorised praw instance to read reddit
posts_dict1 = {"Post_ID": [], "Post_Text": [], "User_Hash": [], "orientation": []}
#initialises pandas dataframe with fields I'm interested in


def scraper(sub, orientation):
        posts = redditRead.subreddit(sub)
        #reads the subreddit
        print("reading sub:", sub) #debugging line, lets me know the loop is running
        for post in posts.hot(limit=1000):  #goes through a maximum of the top 1000 posts on a subreddit, retrieving information
                if post.is_self: #adds post to dataframe if they have body text 
                        AllText = post.title + "\n" + post.selftext #combines post title and text into one variable
                        posts_dict1["Post_Text"].append(AllText)
                        posts_dict1["Post_ID"].append(post.id)
                        posts_dict1["User_Hash"].append(hash(post.author))#hashes the username to stay within ethical guidelines
                        posts_dict1["orientation"].append(orientation)
                        #takes specific fields from the posts, builds list
                        submission = redditRead.submission(id=post.id)
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list():
                                if len(comment.body)>200:
                                        posts_dict1["Post_Text"].append(comment.body)
                                        posts_dict1["Post_ID"].append(post.id)
                                        posts_dict1["User_Hash"].append(hash(comment.author))
                                        posts_dict1["orientation"].append(orientation)
                        #takes text from every top level comment of the post that is over 150 characters and adds to the dataset
                                        




rwsubreddit_list = ["republican", "libertarian","tories","conservative","anarcho_capitalism","trump", "louderwithcrowder"]
lwsubreddit_list = ["communism","socialism","labouruk","greenandpleasant","democrats", "anarchocommunism","anarchism"]

for sub in rwsubreddit_list:
        scraper(sub,"1")
        #loops through list of right wing subreddits i've chosen, and runs the scraper.
        #the parameter '1' designates that these are right wing 

for sub in lwsubreddit_list:
        scraper(sub, "0")
        #does the same but with the list of left wing subreddits, and 0 to designate that
posts_file = pd.DataFrame(posts_dict1)
posts_file.to_csv("posts.csv", mode="a", index=True, header=True)
#appends the data retrieved into a csv file

