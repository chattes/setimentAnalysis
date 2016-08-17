from __future__ import division
import random
import cPickle as pickle
import nltk
import codecs
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode, StatisticsError

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
#Most Common Vote- Mode in Python-Return the most common data point from discrete or nominal data
        try:
            most = mode(votes)
        except StatisticsError:
            print "There is no Mode"
            most = "Unknown"
        return most

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        try:
            choice_votes = votes.count(mode(votes))
            conf = choice_votes / len(votes)
            return conf
        except StatisticsError:
            return 0
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("pickled_algos/word_features.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()
def find_features(documents):
    #Find feature words in reviews positive and negative
    # words = set(documents)
    words = set(nltk.word_tokenize(documents))
    features = {}
    #Creates a Tuple with the Word and If it exisits in the review
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets_f = open("pickled_algos/featureset.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

training_set = featuresets[:10000]
testing_set = featuresets[10000:]
#Save Classifiers for Future
open_file = open("pickled_algos/naivebayes.pickle","rb")
classifier = pickle.load(open_file)
open_file.close()


# open_file = open("pickled_algos/MNB.pickle","rb")
# MNB_classifier = pickle.load(open_file)
# open_file.close()

# open_file = open("pickled_algos/BNB.pickle","rb")
# BNB_classifier = pickle.load(open_file)
# open_file.close()

open_file = open("pickled_algos/logregression.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/SGDC.pickle","rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/SVC.pickle","rb")
SVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/LinearSVC.pickle","rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/NuSVC.pickle","rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
        NuSVC_classifier,
        LogisticRegression_classifier,
        SGDClassifier_classifier,
        SVC_classifier,
        LinearSVC_classifier
        )
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
