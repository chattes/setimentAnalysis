import random
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
#Most Common Vote- Mode in Python-Return the most common data point from discrete or nominal data
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf



stop_words = set(stopwords.words('english'))

documents = [(list(filter(lambda w:not w in stop_words,movie_reviews.words(fileid))),category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)
#Save Document set
save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
all_words = dict(all_words.most_common(5000))


#Top 3000 most common words used in Movie Reviews
word_features = list(all_words.keys())
save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()
def find_features(documents):
    #Find feature words in reviews positive and negative
    words = set(documents)
    features = {}
    #Creates a Tuple with the Word and If it exisits in the review
    for w in word_features:
        features[w] = (w in words)
    return features



featuresets = [(find_features(rev),category) for (rev, category) in documents]
random.shuffle(featuresets)
featureset_save = open("pickled_algos/featureset.pickle","wb")
pickle.dump(featuresets,featureset_save)
featureset_save.close()


training_set = featuresets[:1900]
testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(20)

#Save Classifiers for Future
save_classifier = open("pickled_algos/naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

# #Other Classifiers
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))
# save_classifier = open("pickled_algos/MNB.pickle","wb")
# pickle.dump(MNB_classifier,save_classifier)
# save_classifier.close()



# BNB_classifier = SklearnClassifier(BernoulliNB())
# BNB_classifier.train(training_set)
# print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))
# save_classifier = open("pickled_algos/BNB.pickle","wb")
# pickle.dump(BNB_classifier,save_classifier)
# save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open("pickled_algos/logregression.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
save_classifier = open("pickled_algos/SGDC.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/SVC.pickle","wb")
pickle.dump(SVC_classifier,save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/LinearSVC.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/NuSVC.pickle","wb")
pickle.dump(NuSVC_classifier,save_classifier)
save_classifier.close()



voted_classifier = VoteClassifier(classifier,
                                NuSVC_classifier,
                                # MNB_classifier,BNB_classifier,
                                SGDClassifier_classifier,
                                SVC_classifier,
                                LinearSVC_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
