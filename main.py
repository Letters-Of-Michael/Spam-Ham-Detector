import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


naive_bayes = MultinomialNB()
count_vector = CountVectorizer()



header = st.container()
dataset = st.container()
features = st.container()
Data_Development = st.container()
model_training = st.container()
your_time = st.container()
copywright = st.container()

#st.cache this helps to make our app run faster in case of large data set
def get_data(filename):
	spam_data = pd.read_csv(filename)

	return spam_data


with header:
	st.title('SPAM/HAM DETECTION APP')
	st.markdown('* **Welcome, I am Oluwaseyi Michael. This is my first Machine Learning App**')
	st.write('In this model, I developed a Spam-Ham detector system that classifies a text/email as spam or ham. This model has been trained with various algorithm and the one with the highest accuracy is modelled here, it allows this model to easily clasify these texts to either be a spam; which is an unwanted text or a ham which is a wanted, proper text')


	with dataset:
		st.header('About Dataset')
		st.write('The dataset contains about 5572 email texts collected from our sample, this is a dataset from kaggle and we were able to use it to train our model, exploring the dataset using python on jupyter notebook, we have no null cell, with all data type being object. The dataset has three columns: index, label and message. We have about 4825 messages as Ham and 747 labelled as Spam.')

		spam_data = pd.read_csv('./data/spim.csv', encoding = 'ISO-8859-1')
		st.write(spam_data.head())
		spam_data = spam_data[['label', 'message']]
		#spam_data['label'] = spam_data.label.map({'ham':0, 'spam':1})
		x = np.array(spam_data['message'])
		y = np.array(spam_data['label'])

		st.subheader('Distribution of the dataset')
		label_dist = pd.DataFrame(spam_data['label'].value_counts())
		st.bar_chart(label_dist)


	with features:
		st.header('Features And Text Processing')
		st.write('In this section, I described some features for the dataset. Looking into our dataset you noticed our inputs are all texts and machine learning only deal with numbers, so, I had to convert to number vectors using the CountVectorizer. I had to replace some abbreviations in the messages, convert all to lower case and also common figures converted to text equivalents.')

	with Data_Development:
		st.header('Data Development')
		st.write('In this section, I used the sklearn module to split our data into the training text and testing texts, we split in 70%-30%, then assigned the testing and training data. This is the core aspect of our training in order to develop a perfect model.')



	with model_training:
		st.header('Training the model!')
		st.write('In this section, I trained the Model using the Naives Bayes algorithm from the sklearn module. Our prediction was also run and a perfect prediction was got. I evaluated the model with the sklearn metrics using the Accuracy_score, precision_score, recall_score and f1_score. The model came up with an accuracy of 99%.')

	with your_time:
		st.header('Try in your message here!')
		sel_col, disp_col = st.columns(2)

		X = count_vector.fit_transform(x) #fit data
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

		#spam_data['label'] = spam_data.label.map({'ham':0, 'spam':1})

		training_data = count_vector.fit_transform(x) #fit data

		clf = MultinomialNB()
		clf.fit(X_train,y_train)

		def spamdetection():
			user = sel_col.text_input('Please, type in your message or Email:')
			prediction = st.button('Predict type of message')


			if len(user) < 1:
				st.write(' ')
			else:
				sample = user
				testing_data = count_vector.transform([sample]).toarray()
				prediction = clf.predict(testing_data)
				st.title(prediction)
		spamdetection()





	with copywright:
		st.text('By Oluwaseyi Michael')


