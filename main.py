#Importing Basic Libraries
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# removing twitter handles
def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)
	return input_txt


# function to collect hashtags
def hashtag_extract(x):
	hashtags = []
	# Loop over the words in the tweet
	for i in x:
		ht = re.findall(r"#(\w+)", i)
		hashtags.append(ht)
	return hashtags


def main():
	#Loading Dataset
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	#Combining Dataset for furture assesment
	combining = train.append(test, ignore_index=True)

	#Removing Twitter Handels
	combining['tidy_tweet'] = np.vectorize(remove_pattern)(combining['tweet'], "@[\w]*")

	# Removing Special Chanracters, Punctuations, Number
	combining['tidy_tweet'] = combining['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

	#Removing Short Words
	combining['tidy_tweet'] = combining['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

	#Tokenizing Tweets means splitting into individual words
	tokenized_tweet = combining['tidy_tweet'].apply(lambda x: x.split())

	#Stemming Tweets means to stripping the suffixes
	stemmer = PorterStemmer()
	tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming

	#Combining Tweets back after Tokenzing and Stemming
	for i in range(len(tokenized_tweet)):
		tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
	combining['tidy_tweet'] = tokenized_tweet

	#Generating a WordCloud of all the words present
	all_words = ' '.join([text for text in combining['tidy_tweet']])
	wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
	plt.figure(figsize=(10, 7))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis('off')
	plt.show()

	#Generating WordCloud for all the positive words
	normal_words = ' '.join([text for text in combining['tidy_tweet'][combining['label'] == 0]])
	wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
	plt.figure(figsize=(10, 7))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis('off')
	plt.show()

	#Generating WordCloud for all the negative Words
	negative_words = ' '.join([text for text in combining['tidy_tweet'][combining['label'] == 1]])
	wordcloud = WordCloud(width=800, height=500,
						  random_state=21, max_font_size=110).generate(negative_words)
	plt.figure(figsize=(10, 7))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis('off')
	plt.show()

	#Extracting hashtags from non racist/sexist tweets
	HT_regular = hashtag_extract(combining['tidy_tweet'][combining['label'] == 0])

	#Extracting hashtags from racist/sexist tweets
	HT_negative = hashtag_extract(combining['tidy_tweet'][combining['label'] == 1])

	#Unnesting list
	HT_regular = sum(HT_regular, [])
	HT_negative = sum(HT_negative, [])

	#PLotting Graphs for Top 10 positive words
	a = nltk.FreqDist(HT_regular)
	d = pd.DataFrame({'Hashtag': list(a.keys()),
					  'Count': list(a.values())})
	# selecting top 10 most frequent hashtags
	d = d.nlargest(columns="Count", n=10)
	plt.figure(figsize=(16, 5))
	ax = sns.barplot(data=d, x="Hashtag", y="Count")
	ax.set(ylabel='Count')
	plt.show()

	#PLotting Graphs for top 10 Negative Words
	b = nltk.FreqDist(HT_negative)
	e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
	# selecting top 10 most frequent hashtags
	e = e.nlargest(columns="Count", n=10)
	plt.figure(figsize=(16, 5))
	ax = sns.barplot(data=e, x="Hashtag", y="Count")
	ax.set(ylabel='Count')
	plt.show()

	bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
	# bag-of-words feature matrix
	bow = bow_vectorizer.fit_transform(combining['tidy_tweet'])

	tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
	# TF-IDF feature matrix
	tfidf = tfidf_vectorizer.fit_transform(combining['tidy_tweet'])

	#Model of Bag of Words features
	train_bow = bow[:31962, :]
	test_bow = bow[31962:, :]
	# splitting data into training and validation set
	xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=1, test_size=0.2)
	lreg = LogisticRegression()
	lreg.fit(xtrain_bow, ytrain)  # training the model
	lreg_pred = lreg.predict(xvalid_bow)
	print('accuracy score ', accuracy_score(yvalid, lreg_pred))

	#Model of TF_IDF features
	train_tfidf = tfidf[:31962, :]
	test_tfidf = tfidf[31962:, :]
	xtrain_tfidf = train_tfidf[ytrain.index]
	xvalid_tfidf = train_tfidf[yvalid.index]
	lreg.fit(xtrain_tfidf, ytrain)
	lreg_pred_tfidf = lreg.predict(xvalid_tfidf)
	print('accuracy score ', accuracy_score(yvalid, lreg_pred_tfidf))


main()