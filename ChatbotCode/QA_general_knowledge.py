import string

import nltk

from util import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import joblib

# TODO: load it in from joblib, as the question space is the same every time

questions = []
answers = []


# Save vector space to job lib
def make_QA_vector_space():
    filepath = 'datasets/QA_dataset.csv'
    with (open(filepath, 'r', encoding='utf8') as file):
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            question = ' '.join(preprocess_text(row[0]))
            response = row[1]
            questions.append(question)
            answers.append(response)
    # Make vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Make a term-freq matrix of the questions, and fits vocab to vectorizer
    tfidf_ques_ans = tfidf_vectorizer.fit_transform(questions)
    joblib.dump(tfidf_ques_ans, 'tfidf_question_matrix.joblib')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    print(questions)


def find_similar_q(my_question, threshold):
    # Process User_Input
    tokens = preprocess_text(my_question)
    processed_user_input = " ".join(tokens)

    # Maps onto the same vector space AND tokenizes and tfidf weighs them
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    user_vector = tfidf_vectorizer.transform([processed_user_input])

    tfidf_matrix = joblib.load('tfidf_question_matrix.joblib')
    # Calculate cosine similarity between the search vector and all question vectors
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)

    # Get the index of the most similar question
    most_similar_index = cosine_similarities.argmax()

    # most_similar_indexes = cosine_similarities.argsort()[::-1][:3] # -1 starts from the end, 3 returns array of top
    # Get the most similar question
    print("Most similar question: %s", questions[most_similar_index])
    if cosine_similarities[0, most_similar_index] < threshold:
        print("I don't know the answer to that question.")
        return False
    else:
        print(answers[most_similar_index])
        return True


# TODO: Evaluate the accuracy of the information retrieval system, using lab3
