from util import preprocess_text, make_arrays_from_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# Save vector space to job lib
def make_Q_vector_space():
    questions, labels = make_arrays_from_csv('datasets/QA_dataset.csv')

    # Make vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Make a term-freq matrix of the questions, and fits vocab to vectorizer
    tfidf_ques_ans = tfidf_vectorizer.fit_transform(questions)
    joblib.dump(tfidf_ques_ans, 'tfidf_question_matrix.joblib')
    joblib.dump(tfidf_vectorizer, 'tfidf_qa_vectorizer.joblib')
    return questions

#make_Q_vector_space()
def find_similar_q(my_question, threshold):
    questions, answers = make_arrays_from_csv('datasets/QA_dataset.csv')

    # Process User_Input
    tokens = preprocess_text(my_question)
    processed_user_input = " ".join(tokens)

    # Maps onto the same vector space AND tokenizes and tfidf weighs them
    tfidf_vectorizer = joblib.load('tfidf_qa_vectorizer.joblib')
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
        return False
    else:
        print(answers[most_similar_index])
        return True

# TODO: Evaluate the accuracy of the information retrieval system, using lab3
