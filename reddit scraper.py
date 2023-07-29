#dissertation project subreddit reader
#Robert Philipson


import praw
import pandas as pd
import nltk
import time
start_time = time.time()
redditRead = praw.Reddit()#make your own authorised instance here!
#Creates an authorised praw instance to read reddit
posts_dict1 = {"Post_ID": [], "Post_Text": [], "User_Hash": [], "orientation": [], "sub":[]}
#initialises pandas dataframe with fields I'm interested in


def scraper(sub, orientation, readcount):
                posts = redditRead.subreddit(sub)
                #reads the subreddit
                print("reading sub:", sub) #debugging line, lets me know the loop is running
                for post in posts.hot(limit=1000):  #goes through the top 1000 'hot' posts on a subreddit, retrieving information
                        readcount +=1
                        if (post.is_self or len(post.title)>120) and post.author != "AutoModerator": #adds post to dataframe if they have body text and aren't a bot mod

                                AllText = post.title + "\n" + post.selftext #combines post title and text into one variable
                                posts_dict1["Post_Text"].append(AllText.lower())
                                posts_dict1["Post_ID"].append(post.id)
                                posts_dict1["User_Hash"].append(hash(post.author))#hashes the username to stay within ethical guidelines
                                posts_dict1["orientation"].append(orientation)
                                posts_dict1["sub"].append(sub)
                                #takes specific fields from the posts, builds list
                                submission = redditRead.submission(id=post.id)
                                submission.comments.replace_more(limit=0)
                                for comment in submission.comments.list():
                                        readcount +=1
                                        if len(comment.body)>200 and comment.author != "AutoModerator" and comment.author != "socialism-ModTeam":
                                                posts_dict1["Post_Text"].append((comment.body).lower())
                                                posts_dict1["Post_ID"].append(post.id)
                                                posts_dict1["User_Hash"].append(hash(comment.author))
                                                posts_dict1["orientation"].append(orientation)
                                                posts_dict1["sub"].append(sub)
                                #takes text from every top level comment of the post that is over 150 characters and adds to the dataset
                                #also filters out bot moderator comments





def loops():
        readcount = 0
        rwsubreddit_list = ["republican", "libertarian","tories","conservative","anarcho_capitalism","trump"]
        lwsubreddit_list = ["communism","socialism","labouruk","greenandpleasant","democrats", "anarchocommunism","anarchism"]
        for sub in rwsubreddit_list:
                scraper(sub,"1", readcount)
                #loops through list of right wing subreddits i've chosen, and runs the scraper.
                #the parameter '1' designates that these are right wing 

        for sub in lwsubreddit_list:
                scraper(sub, "0", readcount)
                #does the same but with the list of left wing subreddits, and 0 to designate that
        posts_file = pd.DataFrame(posts_dict1)
        print(readcount)
        posts_file.to_csv("posts.csv", mode="a", index=True, header=True)
        print("--- %s seconds ---" % (time.time() - start_time))

loops()

#appends the data retrieved into a csv file

