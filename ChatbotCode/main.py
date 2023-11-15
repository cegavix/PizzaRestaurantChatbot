from intent_matching import *
from QA_general_knowledge import *
from util import *

# download_nltk_resources()
# TODO: stop using array of intents and make corpus to use
# TODO: Context tracking, u want the chatbot to remember stuff
if __name__ == "__main__":
    user_name = 'User'
    flag = True
    user_input = input("Hi, I'm Papa the Chatbot for Papa's Pizzeria! And what is your name?")
    user_name = set_name(user_input)
    print("Papa: Hello %s, it is nice to meet you! How can I help you today?" % user_name)

    while flag:
        user_input = input('%s: ' % user_name)

        intent = classify_intent_similarity(user_input)
        # small talk, questions intents . compare similarity to a QA to similarity to intent. Could add all questions
        # to intent matcher? Long, processes the questions twice this way.. but joblib could help
        # print("Your intent is:", intent)

        if intent == ['greeting']:
            print("How's it going!")
        elif intent == ['exit']:
            print("Thank you and Arrivederci!")
            break
        elif intent == ['booking']:
            print('Transferring u onto booking personnel...')
        elif intent == ['menu']:
            print('Here\'s the menu:')
        elif find_similar_q(user_input, 0.7):
            print("Hope that answered ur question... next most similar question?")
        # elif small_talk_model()
        else:
            print("I'm not sure I understand!")
