import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize ,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


file_path = 'data'  # Replace with the actual path to your file
with open(file_path, 'r', encoding='utf8') as f:
    data = f.read().replace('\n', ' ')

  # Tokenize the text into sentences
sentences = sent_tokenize(data)
# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('french') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]


# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence


def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence

question= input("Enter your question: ")
response = chatbot(question)
print("Chatbot: " + response)

import streamlit as st
def main():
    st.title("Avis au peuple sur sa santé")
    st.write("Posez-moi des questions sur ce sujet .")
    question = st.text_input("Vous :")
    if st.button("Soumettre"):
        response = chatbot(question)
        st.write("Chatbot : " + response)

if __name__ == "__main__":
    main()

