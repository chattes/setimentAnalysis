from __future__ import division
import sentimentAnalysis as s
from nltk.stem.porter import PorterStemmer
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from pymongo import MongoClient
import urllib
porter = PorterStemmer()


# path = 'http://www.unc.edu/~ncaren/haphazard/'
# files = ['negative.txt','positive.txt']
#
# for filename in files:
#     urllib.urlretrieve(path+filename,filename)
#
# positive_words = open('positive.txt').read().split('\n')
# negative_words = open('negative.txt').read().split('\n')
# # TODO  get the actual Tweets from MongoDB
client = MongoClient()
discoverMovies = client['discovermovies']
tweets = list(discoverMovies.tweetdbs.find({"Rating":None}))
unique_movie = discoverMovies.tweetdbs.distinct("movie_id")
stop_words = set(stopwords.words('english'))

# unique_movie = set()
# for tweet in tweets:
#     unique_movie.add(tweet.get("movie_id"))
#
for movie in unique_movie:
    movie_tweets = list(filter(lambda d: d['movie_id'] == movie,tweets))
    for movie_tweet in movie_tweets:
        tweetProcessed = movie_tweet["text"].lower()
        for p in list(punctuation):
            tweetProcessed = tweetProcessed.replace(p,'')
        print "Processing Tweet....%s" %tweetProcessed
        print "Printing Results..."

        tweet_words = word_tokenize(tweetProcessed)
        
        list_of_words = [porter.stem(w) for w in tweet_words if w not in stop_words]
        print(list_of_words)


        tweetProcessed = " ".join(list_of_words)
        print("Tweet after process"+ tweetProcessed)
        resultSet = s.sentiment(tweetProcessed)
        discoverMovies.tweetdbs.update({"tweetId":movie_tweet["tweetId"]},
                                       {
                                           "$set":{
                                               "Rating":resultSet[0],
                                               "Confidence":resultSet[1]
                                               }
                                           })
        print(resultSet[0])
        print("-------------------------------------")

        # words = tweetProcessed.split(' ')
        # for word in words:
        #     positive_counter = 0
        #     if word in positive_words:
        #         positive_counter = positive_counter + 1
        # print positive_counter/len(words)
        #
