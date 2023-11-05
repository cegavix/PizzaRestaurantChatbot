import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string, re
import random

# class QuestionAnsweringChatbot:
#     def __init__(self, questions, answers):
#         self.vectorizer = TfidfVectorizer()
#         self.questions = questions
#         self.answers = answers
#
#         # Transform the questions and answers into a TF-IDF representation
#         X = self.vectorizer.fit_transform(questions)
#
#     def answer_question(self, question):
#         # Transform the user's question into a TF-IDF representation
#         q = self.vectorizer.transform([question])
#
#         # Calculate the cosine similarity between the user's question and all of the questions in the dataset
#         similarities = np.dot(q, X.T)
#
#         # Find the question with the highest cosine similarity to the user's question
#         most_similar_question_index = np.argmax(similarities)
#
#         # Return the answer to the most similar question
#         return self.answers[most_similar_question_index]


# Preprocess and import data
with open("QA_dataset.csv", encoding='utf8', errors='ignore', mode='r') as document:
    content = document.read().lower()

# Remove stop words from the questions and answers
stop_words = nltk.corpus.stopwords.words("english")
words = content.split()
words = [word for word in words if word not in stop_words]
content = " ".join(words)

sent_tokens = nltk.sent_tokenize(content)  # converts to list of sentences
word_tokens = nltk.word_tokenize(content)

lemmer = nltk.stem.WordNetLemmatizer()

def remove_punctuation(text):
    removed = re.sub(r"[^\w' ]",' ',text) # Remove most punctuation
    return removed
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
    robot_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robot_response=robot_response+"I think I need to read more about that..."
        return robot_response
    else:
        robot_response = robot_response+sent_tokens[idx]
        return robot_response


GREETING_INPUTS = ("hello", "greetings", "hi", "hey", "whats up", "howdy")
GREETING_RESPONSES = ["Hey there!", "What can i do for you", "hello", "hi", "whats up", "hey", "how can i help you"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


flag=True
print("Ask me anything, random knowledge and I'll see what I know....")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ChatBot: Anytime")
        else:
            if(greeting(user_response)!=None):
                print("ChatBot: "+greeting(user_response))
            else:
                print("ChatBot: ",end="")
                print(response(user_response))
                #sent_tokens.remove(user_response)
    else:
        flag=False
        print("ChatBot: Bye!")


