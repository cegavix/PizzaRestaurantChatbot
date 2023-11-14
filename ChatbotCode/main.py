import nltk
from intent_matching import *
from QA_general_knowledge import *


from util import *

download_nltk_resources()
#TODO: stop using array of intents and make corpus to use
# TODO: pickle stuff so that the models for intents blah blah are all done
if __name__ == "__main__":
    user_name = 'User'
    flag = True
    user_input = input("Hi, I'm Papa the Chatbot for Papa's Pizzeria! And what is your name?")
    user_name = set_name(user_input)
    print("Papa: Hello %s, it is nice to meet you! How can I help you today?" %user_name)

    while flag:
        user_input = input('%s: ' %user_name)
        if user_input == 'q':
            flag = False
        intent = classify_input_short(user_input)
        # small talk, questions intents . compare similarity to a QA to similarity to intent.
        # Could add all questions to intent matcher? Long, processes the questions twice this way.. but joblib could help
        print("Your intent is: %s", intent)
        if intent == ['greeting']:
            print("How's it going!")
        if intent == ['booking']:
            print('Transferring u onto booking personnel...')
        if intent == ['menu']:
            print('Heres the menu:')
        find_similar_q(user_input, 0.7)




