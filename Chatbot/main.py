import nltk
from sklearn.metrics.pairwise import cosine_similarity
from intent_matching import *
from nltk import word_tokenize, ne_chunk, pos_tag
import csv
from util import *

download_nltk_resources()

if __name__ == "__main__":
    user_name = 'User'
    flag = True
    user_input = input("Hi, I'm Papa the Chatbot for Papa's Pizzeria! And what is your name?")
    user_name = set_name(user_input)
    print("Papa: Hello %s, it is nice to meet you! How can I help you today?" %user_name)

    while flag:
        user_input = input('%s: ' %user_name)
        #if one word Noun response -> isname
        if calculate_intent(user_input, 0.9, name_intents):
            user_name = set_name(user_input)

        if calculate_intent(user_input,0.5,booking_intents):
            print('Papa: Redirecting you to the booking personnel....')


        if user_input == 'q':
            flag = False


