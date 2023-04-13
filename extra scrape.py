import praw
import pandas as pd
import nltk
redditRead = praw.Reddit(client_id="ZHd4uF6mhAs2e8_rFnJLnA",
                                client_secret="OlnDy_ZTJLX9CHXXGnJ4-3Z-fRbFOg",
                                user_agent="Dissertation Scraper",
                                username="Dissbot2000",
                                password="MEGAsecretdisspass2022")
#Creates an authorised praw instance to read reddit
posts_dict1 = {"Post_Text": []}
#initialises pandas dataframe with fields I'm interested in


def scraper(sub):
        posts = redditRead.subreddit(sub)
        #reads the subreddit
        print("reading sub:", sub) #debugging line, lets me know the loop is running
        for post in posts.hot(limit=1000):  #goes through the top 1000 'hot' posts on a subreddit, retrieving information
                if (post.is_self or len(post.title)>120) and post.author != "AutoModerator": #adds post to dataframe if they have body text and aren't a bot mod
                        AllText = post.title + "\n" + post.selftext #combines post title and text into one variable
                        posts_dict1["Post_Text"].append(AllText.lower())
                        #takes specific fields from the posts, builds list
                        submission = redditRead.submission(id=post.id)
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list():
                                if len(comment.body)>200 and comment.author != "AutoModerator" and comment.author != "socialism-ModTeam":
                                        posts_dict1["Post_Text"].append((comment.body).lower())
                        #takes text from every top level comment of the post that is over 150 characters and adds to the dataset
                        #also filters out bot moderator comments
                                        




scraper('republican')
posts_file = pd.DataFrame(posts_dict1)
posts_file.to_csv("extraposts.csv", mode="a", index=True, header=True)
#appends the data retrieved into a csv file
