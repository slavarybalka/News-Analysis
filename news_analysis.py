#newsAPI: xxxxx
# Step 1 retrieve the top news from the source you need every 5 mins
# Step 2 identify in string is in news article
# Step 3 run sentiment analysis on the article

import requests
import json
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
def word_feats(words):
    return dict([(word, True) for word in words])
 

try:
	r = requests.get("https://newsapi.org/v1/articles?source=cnn&apiKey=2e04e5745b78465c8dff44db8d93b1fe")
	#print article titles
	for i in r.json()['articles']:
		print 'Title:', i['title']
		print 'Description:', i['description'], '\n'
		print word_feats(i['title'])		
except:
	print "some error occured"
	pass


negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
 
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()


