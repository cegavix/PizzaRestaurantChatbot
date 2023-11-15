import nltk, re, pprint, string
from collections import Counter
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams, bigrams
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

def process(user_input):
    # Processing input includes tokenising so that it can be readable 
    tokens = nltk.word_tokenize(user_input)
    return tokens

# Main IO loop,ask user for input, process it, output the result back. exit when 'exit' is type

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit': # if the input made lower case is exit
            print("Goodbye!")
            break
        processed_input = process(user_input)
        print(f"You said: {processed_input}")

if __name__ == "__main__":
    main()