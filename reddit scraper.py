#dissertation project subreddit reader
#Robert Philipson


import praw
import pandas as pd
redditRead = praw.Reddit(client_id="ZHd4uF6mhAs2e8_rFnJLnA",
                                client_secret="OlnDy_ZTJLX9CHXXGnJ4-3Z-fRbFOg",
                                user_agent="Dissertation Scraper",
                                username="Dissbot2000",
                                password="MEGAsecretdisspass2022")
#Creates an authorised praw instance to read reddit
posts_dict1 = {"Post_ID": [],"Title": [], "Post_Text": [], "User_Hash": [], "orientation": []}
#initialises pandas dataframe


def scraper(sub, orientation):
        posts = redditRead.subreddit(sub)
        #reads the subreddit
        print("reading sub:", sub) #debugging line, lets me know the loop is running
        for post in posts.hot(limit=1000):  #goes through a maximum of the top 1000 posts on a subreddit, retrieving information
                posts_dict1["Title"].append(post.title)
                posts_dict1["Post_Text"].append(post.selftext)
                posts_dict1["Post_ID"].append(post.id)
                posts_dict1["User_Hash"].append(hash(post.author))#hashes the username to stay within ethical guidelines
                posts_dict1["orientation"].append(orientation)
        #takes specific fields from the posts, builds list




rwsubreddit_list = ["republican", "libertarian","tories","conservative","anarcho_capitalism","trump"]
lwsubreddit_list = ["communism","socialism","labouruk","greenandpleasant","democrats"] 
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

