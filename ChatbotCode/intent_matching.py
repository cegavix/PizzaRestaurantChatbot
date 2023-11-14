import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from HumanAIChatBot.ChatbotCode.util import preprocess_text


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
def classify_input(user_response_of_intent):
    # Takes a split data set... why? This is for training testing purposes only.
    # Should be able to take multiple lines with a \n in the middle, just does the firstl ine(see split)
    # Make the bag of words as a CountVectorizer, and then fit training data on and transform to term-freq
    # Count Vectorizer.fit_transform creates an array (the width of the vocab) with a count for how much appears

    # Read in content from csv
    file_path = 'datasets/intentmatch_dataset.csv'
    # Open the CSV file
    with (open(file_path, 'r', encoding='utf8') as file):
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Skip header
        next(csv_reader)
        for row in csv_reader:
            question = row[0]
            intent = row[1]
            data.append(question)
            labels.append(intent)

    count_vect = CountVectorizer()
    response_counts = count_vect.fit_transform(data)
    # Make weighting transformer: Make a weighted tfidf transformer to apply to the training data after
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(response_counts)
    # Transform: Convert the count training data into tfidf weighted format
    response_tf = tfidf_transformer.transform(response_counts)
    # Make model with tfidf weighted term-freq training data

    # Do this for now, despite making loads of joblibs
    joblib.dump(count_vect, 'count_intentvectorizer.joblib')
    joblib.dump(tfidf_transformer, 'tfidf_intenttransformer.joblib')
    classifier = LogisticRegression(random_state=0).fit(response_tf, labels)

    # Assuming tfidf_vectorizer is the object you want to save
    joblib.dump(classifier, 'intent_classifier.joblib')


    # Model done. Now predict:
    # Process users response, using the same transformer and vectorizer as earlier
    # Count frequency of vocab: Fit
    x_user_counts = count_vect.transform(user_response_of_intent.split('/n'))
    # Transform using Weight
    x_user_tf = tfidf_transformer.transform(x_user_counts)



    return classifier.predict(x_user_tf)



def classify_input_short(user_response_of_intent):
    # Loads the joblibs, preprocess user_response, decides if response is strong enough
    count_vect = joblib.load('count_intentvectorizer.joblib')
    tfidf_transformer = joblib.load('tfidf_intenttransformer.joblib')
    #to do: use the algorithm from the long, make sure short as possible
    # Load classifier
    classifier = joblib.load('intent_classifier.joblib')
    x_user_counts = count_vect.transform(user_response_of_intent.split('/n'))
    # Transform using Weight
    x_user_tfidf = tfidf_transformer.transform(x_user_counts)
    cosine_similarities = cosine_similarity(x_user_tfidf, tfidf_matrix)

    pred_intent = classifier.predict(x_user_tfidf)
    if pred_intent == ['greeting'] and calculate_intent(user_response_of_intent, 0.9, data):
        return pred_intent
    if pred_intent == ['menu'] and calculate_intent(user_response_of_intent, 0.6, data):
        return pred_intent
    if pred_intent == ['booking'] and calculate_intent(user_response_of_intent, 0.5, data):
        return pred_intent
    else:
        return 'NOT FOUND'


print(classify_input_short('i want to book'))

calculate_cosine_similarity()