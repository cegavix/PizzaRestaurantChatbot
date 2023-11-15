import csv
import string
import nltk
from nltk.corpus import stopwords


def download_nltk_resources():
    # Download resources if necessary
    # Try except : catch the errors rather than print to the user
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/wordnet": "wordnet",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "corpora/stopwords": "stopwords"
    }
    for resource_path, resource_id in resources.items():
        try:
            # Check if the resource is available, and download if it is not
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_id}")
            nltk.download(resource_id)


def make_arrays_from_csv(filepath):
    questions = []
    answers = []
    # filepath = 'datasets/QA_dataset.csv'
    # file_path = 'datasets/intentmatch_dataset.csv'
    with (open(filepath, 'r', encoding='utf8') as file):
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            question = ' '.join(preprocess_text(row[0]))
            response = row[1]
            questions.append(question)
            answers.append(response)
    return questions, answers


def preprocess_text(text):
    # Generates lemmatized tokens, free of punctuation and stop words
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords: Seems to keep ruining stuff
    # tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    return tokens


def set_name(my_input):
    # TODO: If u type hello, it will take this as the name, or just the first word!!! fix? ne_chunks? ngrams? (look for (name, NNP) tuples
    # Use library to perform entity recognition using bag of words tagging
    name = "NOT FOUND"

    # Pinpoint which word is the name in input:
    pos_tags = nltk.pos_tag(nltk.word_tokenize(my_input))
    for entity in pos_tags:
        if isinstance(entity, tuple) and entity[1] == 'NNP':  # NNP: Proper noun, singular
            name = entity[0]
            #print("NNP Name found: %s" % entity[0])
            # Once ideal name is found, the system stops looking
            return name
        elif isinstance(entity, tuple) and entity[1] == 'NN':
            name = entity[0]
            #print("Name found: %s" % entity[0])
        elif len(pos_tags) == 1:
            name = entity[0]
    return name

