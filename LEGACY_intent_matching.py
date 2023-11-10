# Let's write a simple intent matching function using NLTK in Python for a restaurant booking system.

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a set of intents
intents = {
    "greeting": ["hello", "hi", "greetings", "sup", "what's up"],
    "booking": ["book", "make a reservation", "reserve", "booking"],
    "cancellation": ["cancel", "cancel reservation", "cancel booking"],
    "menu": ["show menu", "menu", "what do you have", "what's on the menu", "what is there on the menu", "what food "
                                                                                                         "is there",
             "what specials are there", "what food do you have"],
}


# Function to preprocess and tokenize input text
def preprocess_text(text):
    # Convert to lower case
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Function to find the intent of a message
def find_intent(message):
    # Preprocess the message
    tokens = preprocess_text(message)
    # Tag the tokens with part of speech
    pos_tags = nltk.pos_tag(tokens)

    # Calculate the score for each intent based on word matches
    intent_scores = {intent: 0 for intent in intents}
    for word, pos in pos_tags:
        if pos in ['VB', 'NN']:  # Consider only verbs and nouns for intent matching
            for intent, keywords in intents.items():
                if lemmatizer.lemmatize(word) in keywords:
                    intent_scores[intent] += 1

    # Find the intent with the highest score
    best_intent = max(intent_scores, key=intent_scores.get)

    return best_intent, intent_scores




# Example usage
print("Hi Im here to help you make a reservation. How are you and how can i help you today?")
message = "Hi, I'd like to make a booking for two at 7 pm today."
intent, scores = find_intent(message)
print(intent, scores)
