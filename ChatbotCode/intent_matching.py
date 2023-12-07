import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from util import make_arrays_from_csv
from QA_general_knowledge import find_similar_q

# Modules for Calculating Intent using vectoriser cosine similarity
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


def make_vector_space_with_Classifier():
    """
    Makes the vectorizers for smalltalk and intents in joblibs since they are unchanging. Makes the classifier for
    intents.
    """
    # Only needs to be run once (unless datasets are changed). Stop words make the questions useless???
    st_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    intent_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)

    # Make Classifier
    responses, intents = make_arrays_from_csv('datasets/intentmatch_dataset.csv')
    intent_tfidf_matrix = intent_tfidf_vectorizer.fit_transform(responses)
    classifier = LogisticRegression(random_state=0).fit(intent_tfidf_matrix, intents)

    # Dump data structures
    joblib.dump(intent_tfidf_matrix, 'intent_matrix.joblib')
    joblib.dump(st_vectorizer, 'st_vectorizer.joblib')
    joblib.dump(intent_tfidf_vectorizer, 'intent_vectorizer.joblib')
    joblib.dump(classifier, 'intent_classifier.joblib')




def classify_intent_similarity(user_response_of_intent):
    # TODO: Add responses to all the intents, and print them before passing to booking function? why is small talk handled only here and all other intents arent? is it cos no context tracking needed?
    """
    Handles the dynamic aspects of vectorizers, that rely on user input
    Uses both Cosine Similarity and a Classifier to determine sure intent. SMALL TALK is a separate Vector Space and
    uses only cosine similarity. Other intents are in a Vector space together and use a Classifier AND Cosine Similarity.

    :param user_response_of_intent: str, what the user inputted
    :return: Provides the intent, which can be any of the following: SMALL TALK, name, booking, menu, exit
    """

    make_vector_space_with_Classifier()
    # Load in vectorizers
    intent_vectorizer = joblib.load('intent_vectorizer.joblib')
    st_vectorizer = joblib.load('st_vectorizer.joblib')

    # Load/Make term frq matrices
    tfidf_matrix = joblib.load('intent_matrix.joblib')
    st_prompt, st_responses = make_arrays_from_csv('datasets/smalltalk_dataset.csv')
    st_matrix = st_vectorizer.fit_transform(st_prompt)

    # Map on input vectors
    user_vector = intent_vectorizer.transform([user_response_of_intent])
    st_user_vector = st_vectorizer.transform([user_response_of_intent])

    # Predict the intent using classifier
    classifier = joblib.load('intent_classifier.joblib')
    pred_intent = classifier.predict(user_vector)

    # Calculate cosine similarity between the input vector and intent vectors and small talk
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)
    st_cosine_similarities = cosine_similarity(st_user_vector, st_matrix)

    # Get the highest cosine similarity for intent
    most_similar_index = cosine_similarities.argmax()
    highest_similarity = cosine_similarities[0, most_similar_index]

    # Get the highest cosine similarity for small talk
    st_most_similar_index = st_cosine_similarities.argmax()
    st_highest_similarity = st_cosine_similarities[0, st_most_similar_index]

    # find_similar_q(user_response_of_intent)
    # print('Before IFs, i think intent is: %s with a cosine of %f' % (pred_intent, highest_similarity))
    # TODO: Make sure the similarity is for the SAME INTENT!! cos theyre traineddifferent
    # TODO: Maybe I should include some form of 'keywords weights', use stop words to find keywords
    # If small talk cosine is higher than intent cosine:
    if st_highest_similarity > highest_similarity and st_highest_similarity > 0.7:
        best_st_response = st_responses[st_most_similar_index]
        print("Papa:", best_st_response)
        return 'SMALL TALK'
    if pred_intent == ['name'] and highest_similarity > 0.6:
        return pred_intent
    if pred_intent == ['menu'] and highest_similarity > 0.6:
        return pred_intent
    if pred_intent == ['booking'] and highest_similarity > 0.6:
        return pred_intent
    if pred_intent == ['exit'] and highest_similarity > 0.7:
        return pred_intent
    else:
        return 'NOT FOUND'

