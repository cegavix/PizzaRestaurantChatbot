from intent_matching import classify_intent_similarity
from QA_general_knowledge import find_similar_q
from util import *
from transaction_management import make_booking

# download_nltk_resources()
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

        if intent == 'SMALL TALK':
            pass
        elif intent == ['exit']:
            print("Thank you and Arrivederci!")
            break
        elif intent == ['name']:
            change_name = input('Papa: Your name is %s. Would you like to change it? y/n' % user_name).lower()
            if change_name == 'y':
                user_name = set_name(input('What is your name?'))
            print('Okay, %s. What do you want to talk about now?' % user_name)
        elif intent == ['booking']:
            make_booking(user_input, user_name)
        elif intent == ['menu']:
            print('Here\'s the menu:')
            print("\n".join([
                "Margherita Pizza: Classic pizza with tomato, fresh mozzarella, and basil. - $10.99",
                "Spaghetti Bolognese: Spaghetti pasta with a rich meat sauce. - $12.99",
                "Caprese Salad: Fresh tomatoes, mozzarella, and basil drizzled with balsamic glaze. - $8.99"
            ]))
        elif find_similar_q(user_input):
            pass
            # user_input = input("Hope that answered ur question... Did it?")
            # if user_input == 'positive'
            # add(user_input,answer) to dataset
            # Maybe try
        else:
            print("I'm not sure I understand!")
