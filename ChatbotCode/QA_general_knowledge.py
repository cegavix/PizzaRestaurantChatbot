from util import preprocess_text, make_arrays_from_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# Save vector space to job lib
def make_Q_vector_space():
    questions, labels = make_arrays_from_csv('datasets/QA_dataset.csv')

    # Make vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),lowercase=True, stop_words='english')
    # Make a term-freq matrix of the questions, and fits vocab to vectorizer
    ques_matrix = tfidf_vectorizer.fit_transform(questions)
    joblib.dump(ques_matrix, 'question_matrix.joblib')
    joblib.dump(tfidf_vectorizer, 'qa_vectorizer.joblib')


# TODO: Why does it just have absolutely 0 idea ab certain questions, even without stop words? ie. 'what are stocks and bonds'? Ask chatgpt, give her my code

def find_similar_q(my_question):
    # Maps onto the same vector space AND tokenizes and tfidf weighs them
    tfidf_vectorizer = joblib.load('qa_vectorizer.joblib')
    user_vector = tfidf_vectorizer.transform([my_question])

    tfidf_matrix = joblib.load('question_matrix.joblib')

    # Calculate cosine similarity between the search vector and all question vectors
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)
    # Get the index of the most similar question
    most_similar_index = cosine_similarities.argmax()
    print('QA similarity:', cosine_similarities[0, most_similar_index])
    # most_similar_indexes = cosine_similarities.argsort()[::-1][:3] # -1 starts from the end, 3 returns array of top
    # Get the most similar question
    # print("Most similar question:", questions[most_similar_index])
    if cosine_similarities[0, most_similar_index] < 0.5:
        return False
    else:
        questions, answers = make_arrays_from_csv('datasets/QA_dataset.csv')
        print(answers[most_similar_index])
        return True


# TODO: Evaluate the accuracy of the information retrieval system, using lab3
# while True:
#     user_input = input("What is ur q?")
#     print(find_similar_q(user_input))
