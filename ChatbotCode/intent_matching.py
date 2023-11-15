import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from util import make_arrays_from_csv


# Modules for Calculating Intent using vectorisr cosine similarity
# Determine whether a response is similar enough to a sample Response in the dataset
def calculate_intent(query, threshold, intents):
    # See if query matches an intent with TfiDf vectoriser

    tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
    x_counts = tfidf.fit_transform(intents)
    # Y_counts = tfidf.get_feature_names_out(x_counts)

    input_tfidf = tfidf.transform([query.lower()]).toarray()
    cosine_similarities = cosine_similarity(input_tfidf, x_counts)
    if cosine_similarities.max() >= threshold:
        return True
    else:
        return False


data = []
labels = []


# Determine which intent the response matches to with a classifier. Has to be done instantly
def make_vector_space_with_Classifier():
    # Takes a split data set... why? This is for training testing purposes only.
    # Should be able to take multiple lines with a \n in the middle, just does the firstl ine(see split)
    # Make the bag of words as a CountVectorizer, and then fit training data on and transform to term-freq
    # Count Vectorizer.fit_transform creates an array (the width of the vocab) with a count for how much appears
    responses, intents = make_arrays_from_csv('datasets/intentmatch_dataset.csv')
    tfidf_vectorizer = TfidfVectorizer()
    # Make a term-freq matrix of the questions, and fits vocab to vectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(responses)
    joblib.dump(tfidf_matrix, 'tfidf_intent_matrix.joblib')
    joblib.dump(tfidf_vectorizer, 'tfidf_intent_vectorizer.joblib')

    classifier = LogisticRegression(random_state=0).fit(tfidf_matrix, intents)
    # Assuming tfidf_vectorizer is the object you want to save
    joblib.dump(classifier, 'intent_classifier.joblib')


# make_vector_space_with_Classifier()


def classify_intent_similarity(user_response_of_intent):
    tfidf_vectorizer = joblib.load('tfidf_intent_vectorizer.joblib')
    user_vector = tfidf_vectorizer.transform([user_response_of_intent])

    tfidf_matrix = joblib.load('tfidf_intent_matrix.joblib')

    # Predict the intent
    classifier = joblib.load('intent_classifier.joblib')
    pred_intent = classifier.predict(user_vector)

    # Calculate cosine similarity between the search vector and all question vectors
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)

    # Get the index of the most similar question
    most_similar_index = cosine_similarities.argmax()
    highest_similarity = cosine_similarities[0, most_similar_index]

    # TODO: Play around with threshold see which words for the dataset
    if pred_intent == ['greeting'] and highest_similarity > 0.6:
        return pred_intent
    if pred_intent == ['menu'] and highest_similarity > 0.6:
        return pred_intent
    if pred_intent == ['booking'] and highest_similarity > 0.5:
        return pred_intent
    if pred_intent == ['exit'] and highest_similarity > 0.6:
        return pred_intent
    else:
        return 'NOT FOUND'
