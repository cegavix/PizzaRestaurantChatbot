from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
