import os
import sys 
import pandas as pd
import numpy as np
import re
import pyodbc
import datetime
import calendar
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
warnings.filterwarnings("ignore", "SettingWithCopyWarning")

from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

print("Entering....")

#def twitterAnalytics():

def DB_Connection(table_name):
	server = 'innovationstcsunited.database.windows.net'
	database = 'INNOVATIONS_TCSUNITED'
	username = 'united_admin'
	password = 'tcs@1234'
	driver= 'ODBC Driver 13 for SQL Server'

	#Opening Connection
	cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
	cursor = cnxn.cursor()

	#Select
	query = "SELECT * FROM "+table_name;
	df = pd.read_sql(query, cnxn)

	#Closing Connection
	cnxn.close()
	return df

def DBConnection(table_name, column_name, latest_timestamp):
	server = 'innovationstcsunited.database.windows.net'
	database = 'INNOVATIONS_TCSUNITED'
	username = 'united_admin'
	password = 'tcs@1234'
	driver= 'ODBC Driver 13 for SQL Server'

	#Opening Connection
	cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
	cursor = cnxn.cursor()

	query = "SELECT * FROM "+table_name+" WHERE "+column_name+" > "+latest_timestamp;
	df = pd.read_sql(query, cnxn)

	#Closing Connection
	cnxn.close()

	return df

latest_timestamp =  str((calendar.timegm(time.gmtime()) * 1000) - 3600000) #'1543309123000'

df = DBConnection("tweet_data","timestamp_ms", latest_timestamp)
df1 = DB_Connection("timeline_data")

print("DB Done") 

tweets = df
reply = df1

tweets['in_reply_to_tweet_id'] = tweets['tweet_id']

total_tweet = pd.merge(tweets,
				 reply[['tweet_CreatedTime', 'tweet_id', 'tweet_text', 'in_reply_to_tweet_id',
	   'in_reply_to_user_id', 'in_reply_to_screen_name', 'user_id',
	   'user_name', 'screen_name', 'location', 'followers', 'followings',
	   'total_tweets_count', 'account_createdTime', 'retweet_count',
	   'favorite_count', 'verified', 'favourites']],
				 on ='in_reply_to_tweet_id',
				 how ='outer')

print("MergeDone")

airline = DB_Connection('airline_britto')

airline_df = airline

airline = airline.set_index('airlines_name')['airlines_tag'].to_dict()

if 'Nordic Airways' in airline: 
	del airline['Nordic Airways'] #removing fake airways


total_tweet_df = total_tweet

total_tweet.columns

total_tweet['Retweet_Check'] = np.where(total_tweet['tweet_text_x'].str.contains(("RT ")), 1 , 0)

total_tweet['airlines'] = ""

for key, value in airline.items():
	total_tweet['airlines'] = np.where(total_tweet['tweet_text_x'].str.contains(value), key, total_tweet['airlines'])
	total_tweet['airlines'] = np.where(total_tweet['tweet_text_x'].str.contains(key, case = False), key, total_tweet['airlines'])

total_tweet['tweet_id_y'] = total_tweet['tweet_id_y'].fillna("No Reply")
total_tweet['location_x']  = total_tweet['location_x'].fillna("Location Blocked")
total_tweet['location_y']  = total_tweet['location_y'].fillna("Location Blocked")

total_tweet.rename(columns = {'airlines': 'Airlines_Name', 'user_name_x': 'user_name', 'tweet_text_x': 'user_tweet','tweet_text_y': 'reply_tweet', 'tweet_CreatedTime_x': 'user_tweet_created','tweet_CreatedTime_y': 'reply_tweet_created','followers_x':'user_followers','followers_y':'Airlines_followers'}, inplace = True)

total_tweet[['tweet_id_x','in_reply_to_tweet_id']] = total_tweet[['tweet_id_x','in_reply_to_tweet_id']].astype("str")
total_tweet['user_tweet_created'] = pd.to_datetime(total_tweet['user_tweet_created'], errors='coerce')

total_tweet['reply_tweet_created'] = pd.to_datetime(total_tweet['reply_tweet_created'], errors='coerce')

# Replied Tweets & Non-Replied Tweets
total_tweet['reply_status'] = np.where(total_tweet['tweet_id_y'] == "No Reply", "Not_Replied","Replied")

# Ananysing All the Tweets without sentiment analysis - adding sentiment result with replied tweets

#Run one Time otherwise replication will be there.
dummy = pd.get_dummies(total_tweet['reply_status'])
total_tweet = pd.concat([total_tweet, dummy], axis = 1)

airline_df = airline_df.rename(columns = {'airlines_name':'Airlines_Name','airlines_tag':'Airlines_ScreenName'}) #Renaming the column name for merging

total_tweet = pd.merge(total_tweet, airline_df[['Airlines_Name','Airlines_ScreenName']], on='Airlines_Name', how = 'left')

# We could not find the nearest airport to user, ie: tweets doesn't contains the airport codes/ airport names.
total_tweets_with_airlines_df = total_tweet[total_tweet['Airlines_Name'] != "0"] #selecting only airline name df

# Selecting TWEET data alone
total_tweets_with_airlines_tweet_df = total_tweets_with_airlines_df[total_tweets_with_airlines_df['tweet_id_x'] !="nan"]

total_tweets_with_airlines_tweet_df['Response_Time'] = total_tweets_with_airlines_tweet_df['reply_tweet_created'] - total_tweets_with_airlines_tweet_df['user_tweet_created']

#Selecting TIMELINE data alone
total_tweets_with_airlines_timelines_df = total_tweets_with_airlines_df[total_tweets_with_airlines_df['tweet_id_x'] =="nan"]

tweet_df = total_tweets_with_airlines_tweet_df

tweet_df['Response_Time_Seconds'] = tweet_df['Response_Time'].astype('timedelta64[s]')
tweet_df['Response_Time_Minutes'] = tweet_df['Response_Time'].astype('timedelta64[m]')

df = tweet_df #For Backup the Tweet dataframe

print("Before Sentiment Analysis")

# Adding Sentiment Analysis

English_Data = df[['user_tweet']]
English_Data = English_Data.dropna(subset=['user_tweet'])

sid = SentimentIntensityAnalyzer()

se = [] #Empty array

sentValue = []

for index, row in English_Data.iterrows():

	reviewSentence = str(row["user_tweet"]) #selecting comments from each row

	reviewSentence = re.sub('[^ a-zA-Z]', '', reviewSentence) #Removing special characters form the sentence

	ss = sid.polarity_scores(reviewSentence)

	#defining the result of each description
	sentiment = ""

	value = ""

	if ss['compound'] == 0:
		sentiment = "Neutral"
		value = ss['compound']

	elif ss['compound'] > 0:
		sentiment = "Positive"
		value = ss['compound']

	else:
		sentiment = "Negative"
		value = ss['compound']

	#adding values into array
	se.append(sentiment)
	sentValue.append(value)

English_Data['Sentiment'] = se
English_Data['SentimentValue'] = sentValue

df['Sentiment'] = se
df['SentimentValue'] = sentValue

df = df.dropna(axis=0, subset=['Airlines_Name'])

df['Positive'] = np.where(df['Sentiment'] == "Positive", '1', '0')
df['Negative'] = np.where(df['Sentiment'] == "Negative", '1', '0')
df['Neutral'] = np.where(df['Sentiment'] == "Neutral", '1', '0')

df[['Retweet_Check','Not_Replied', 'Replied','Response_Time_Seconds','Response_Time_Minutes','SentimentValue']] = df[['Retweet_Check','Not_Replied', 'Replied','Response_Time_Seconds','Response_Time_Minutes','SentimentValue']].astype(str)

df = df.fillna(0)

a = df #for backup

a['SentimentValue'] = a['SentimentValue'].fillna(0)

a['Response_Time_Minutes'] = np.where(a['Response_Time_Minutes'] == 0, '525600', a['Response_Time_Minutes'])
a['Response_Time_Seconds'] = np.where(a['Response_Time_Seconds'] == 0, '31536000', a['Response_Time_Seconds'])

a['Response_Time_Minutes'] = a['Response_Time_Minutes'].fillna(525600)
a['Response_Time_Seconds'] = a['Response_Time_Seconds'].fillna(31536000)

a['Response_Time_Minutes'] = np.where(a['Response_Time_Minutes'] == 'nan', '525600', a['Response_Time_Minutes'])
a['Response_Time_Seconds'] = np.where(a['Response_Time_Seconds'] == 'nan', '31536000', a['Response_Time_Seconds'])

# Cleaning data
# remove special characters, numbers, punctuations
a['cleaned_tweet'] = a['user_tweet'].str.replace("[^a-zA-Z#]", " ")

a['cleaned_tweet'] = a['cleaned_tweet'].astype('str')

tokenized_tweet = a['cleaned_tweet'].apply(lambda x: x.split())

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

for i in range(len(tokenized_tweet)):
	tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

a['cleaned_tweet_new'] = tokenized_tweet

a['cleaned_tweet_new'] = a['cleaned_tweet_new'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

print("Entering Insert")

b = a

b = b.astype({"user_followers": int, "followings_x": int,"total_tweets_count_x": int, "reply_count": int,"retweet_count_x": int, "favorite_count_x": int,"Airlines_followers": int,"followings_y": int,"total_tweets_count_y": int, "retweet_count_y": int,"favorite_count_y": int, "Retweet_Check": int,"Not_Replied": int, "Replied": int,"Response_Time_Seconds": float, "Response_Time_Minutes": float,"SentimentValue": float, "Positive": int,"Negative": int,"Neutral": int,"Response_Time":str})
b['reply_tweet_created'] = pd.to_datetime(b['reply_tweet_created'], errors='coerce')
b['reply_tweet_created'] = b['reply_tweet_created'].fillna("1970-01-01 00:00:00")

# # Insert data into DB
def insert_to_db(connection, cursor, dataframe):
	for index,row in dataframe.iterrows():
		cursor.execute("INSERT INTO finalTwitTable3([user_tweet_created],[tweet_id_x],[user_tweet],[in_reply_to_tweet_id],[in_reply_to_user_id_x],[in_reply_to_screen_name_x],[user_id_x],[user_name],[screen_name_x],[location_x],[user_followers],[followings_x],[total_tweets_count_x],[account_createdTime_x],[reply_count],[retweet_count_x],[favorite_count_x],[timestamp_ms],[verified_x],[favourites_x],[reply_tweet_created],[tweet_id_y],[reply_tweet],[in_reply_to_user_id_y],[in_reply_to_screen_name_y],[user_id_y],[user_name_y],[screen_name_y],[location_y],[Airlines_followers],[followings_y],[total_tweets_count_y],[account_createdTime_y],[retweet_count_y],[favorite_count_y],[verified_y],[favourites_y],[Retweet_Check],[Airlines_Name],[reply_status],[Not_Replied],[Replied],[Airlines_ScreenName],[Response_Time_Seconds],[Response_Time_Minutes],[Sentiment],[SentimentValue],[Positive],[Negative],[Neutral],[cleaned_tweet]) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
					   row['user_tweet_created'],row['tweet_id_x'],row['user_tweet'],row['in_reply_to_tweet_id'],row['in_reply_to_user_id_x'],row['in_reply_to_screen_name_x'],row['user_id_x'],row['user_name'],row['screen_name_x'],row['location_x'],row['user_followers'],row['followings_x'],row['total_tweets_count_x'], row['account_createdTime_x'],row['reply_count'],row['retweet_count_x'],row['favorite_count_x'],row['timestamp_ms'],row['verified_x'],row['favourites_x'],row['reply_tweet_created'],row['tweet_id_y'],row['reply_tweet'],row['in_reply_to_user_id_y'],row['in_reply_to_screen_name_y'],row['user_id_y'], row['user_name_y'],row['screen_name_y'],row['location_y'],row['Airlines_followers'],row['followings_y'],row['total_tweets_count_y'],row['account_createdTime_y'],row['retweet_count_y'],row['favorite_count_y'],row['verified_y'],row['favourites_y'],row['Retweet_Check'],row['Airlines_Name'],row['reply_status'],row['Not_Replied'],row['Replied'],row['Airlines_ScreenName'],row['Response_Time_Seconds'],row['Response_Time_Minutes'],row['Sentiment'],row['SentimentValue'],row['Positive'],row['Negative'],row['Neutral'],row['cleaned_tweet_new']) 
		connection.commit()

def insertDBConnection(df):
	server = 'innovationstcsunited.database.windows.net'
	database = 'INNOVATIONS_TCSUNITED'
	username = 'united_admin'
	password = 'tcs@1234'
	driver= 'ODBC Driver 13 for SQL Server'

	#Opening Connection
	cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
	cursor = cnxn.cursor()

	#Insert
	insert_to_db(cnxn, cursor, df)

	#Closing Connection
	cnxn.close()

	return df

df = insertDBConnection(b)
print("over")


#scheduler.add_job(twitterAnalytics, 'interval', seconds=2700000)
#scheduler.start()


# In[6]:


#str((calendar.timegm(time.gmtime()) * 1000) - 2700000)

