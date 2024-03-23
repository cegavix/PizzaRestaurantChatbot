from intent_matching import classify_intent_similarity
from util import *
from menu import *
from transaction_management import make_booking

# Un-comment to download necessary nltk resources.
# download_nltk_resources()

if __name__ == "__main__":
    user_name = 'User'
    flag = True
    user_input = input("Hi, I'm Papa the Chatbot for Papa's Pizzeria! And what is your name?")
    user_name = set_name(user_input)
    print("Papa: Hello %s, it is nice to meet you! How can I help you today?" % user_name)

    while flag:
        user_input = input('%s: ' % user_name)

        intent = classify_intent_similarity(user_input,user_name)

        if intent == ['small talk']:
            pass
        elif intent == ['exit']:
            print("Thank you and Arrivederci!")
            break
        elif intent == ['name']:
            change_name = input('Papa: Your name is %s. Would you like to change it? Please say Yes or No:' % user_name).lower()
            if change_name == 'yes':
                user_name = set_name(input('What is your name?'))
            print('Papa: Okay, %s. What do you want to talk about now?' % user_name)
        elif intent == ['booking']:
            make_booking(user_input, user_name)
        elif intent == ['menu']:
            print('Papa: So you are interested in the menu!')
            if display_search_dish(user_input) is None:
                print_menu()
        elif intent == ['question']:
            pass
        else:
            # TODO: make an array of generic acknowledgmenets (CUI)
            print("Papa: Thanks for sharing. What do you want to talk about now?")




