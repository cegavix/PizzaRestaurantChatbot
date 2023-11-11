import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


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


# Determine which intent the response matches to with a classifier
def classify_input(response_train_data, intent_train_data, user_response_of_intent):
    # Should be able to take multiple lines with a \n in the middle, just does the firstl ine(see split)
    # Make the bag of words as a CountVectorizer, and then fit training data on and transform to term-freq
    # Count Vectorizer.fit_transform creates an array (the width of the vocab) with a count for how much appears

    data = []
    labels = []
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
    x_train_counts = count_vect.fit_transform(response_train_data)
    # Make weighting transformer: Make a weighted tfidf transformer to apply to the training data after
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(x_train_counts)
    # Transform: Convert the count training data into tfidf weighted format
    x_train_tf = tfidf_transformer.transform(x_train_counts)
    # Make model with tfidf weighted term-freq training data
    classifier = LogisticRegression(random_state=0).fit(x_train_tf, intent_train_data)

    # Model done. Now predict:
    # Process users response, using the same transformer and vectorizer as earlier
    # Count frequency of vocab: Fit
    x_user_counts = count_vect.transform(user_response_of_intent.split('/n'))
    # Transform using Weight
    x_user_tf = tfidf_transformer.transform(x_user_counts)

    return classifier.predict(x_user_tf)

# Make an information retrieval Similarity Matching