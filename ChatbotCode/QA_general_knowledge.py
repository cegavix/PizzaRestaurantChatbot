from util import make_arrays_from_csv, write_to_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# Save vector space to job lib
def make_Q_vector_space():
    questions, labels = make_arrays_from_csv('datasets/QA_dataset.csv')

    # Make vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 3), lowercase=True, max_df=0.01)

    # Make a term-freq matrix of the questions, and fits vocab to vectorizer
    ques_matrix = tfidf_vectorizer.fit_transform(questions)
    joblib.dump(tfidf_vectorizer, 'qa_vectorizer.joblib')


def new_answer_feedback(question, name):
    print("Papa: If you know the answer, I'd love to learn! Would you be willing to type it out? Type Yes or No:")
    confirm = input("%s:" % name).lower()
    if confirm == 'yes':
        print("Papa: Great! What is the answer to your question?")
        new_answer = input("%s:" % name)
        write_to_csv([question, new_answer], 'datasets/QA_dataset.csv')
    else:
        # print("Papa: No problem. What do you want to talk about now?")
        return False


def find_similar_q(my_question, name):
    """
    :param my_question: the user input that will be assessed to see if it matches a question
    :return: bool return as to whether a question was found or not.

    This is a slow function, hence it coming at the end of the intent matching hierarchy.
    """
    # Uncomment if changes are made to datastructures
    # make_Q_vector_space()
    # Maps onto the same vector space AND tokenizes and tfidf weighs them
    questions, answers = make_arrays_from_csv('datasets/QA_dataset.csv')
    tfidf_vectorizer = joblib.load('qa_vectorizer.joblib')
    tfidf_matrix = tfidf_vectorizer.transform(questions)

    user_vector = tfidf_vectorizer.transform([my_question])
    # Calculate cosine similarity between the search vector and all question vectors
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)

    cosine_similarities = cosine_similarities.flatten()
    most_similar_indexes = cosine_similarities.argsort()[::-1][:3]  # -1 starts from the end, 3 returns array of top
    # Get the most similar question, only add to the array if the values are above 0.6

    index_of_possible_matches = []
    for index in most_similar_indexes:
        if cosine_similarities[index] > 0.5:
            index_of_possible_matches.append(index)

    count = len(index_of_possible_matches)

    if count == 0:
        # No matches found.
        return False
    elif my_question == questions[index_of_possible_matches[0]] or count == 1:
        print(answers[most_similar_indexes[0]])
        return True
    # If more than one question, disambiguation occurs.
    elif count > 1:
        # Multiple matches found, or if 1 found, add utterance to dataset?
        for i in index_of_possible_matches:
            print("Answer", i, ":", answers[i])

        print("\nPapa: Which of these answer your question well? Input the Answer's number, or 'No' if none do:")
        match_index = input('%s: ' % name).lower()
        if match_index != 'no':
            write_to_csv([my_question, answers[int(match_index)]], 'datasets/QA_dataset.csv')
        else:
            new_answer_feedback(my_question, name)
        print("Papa: Thank you for your feedback! You learn something new everyday.")
        return True
