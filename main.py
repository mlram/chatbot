# main.py
import openai
import os
from config import OPENAI_API_KEY
from pdf_extraction import extract_text_from_pdf

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Load PDF content (replace with your PDF paths)
pdf_path_1 = "document1.pdf"
pdf_path_2 = "document2.pdf"

pdf_text_1 = extract_text_from_pdf(pdf_path_1)
pdf_text_2 = extract_text_from_pdf(pdf_path_2)

# Preprocess and index PDF texts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_and_index(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = preprocess_and_index([pdf_text_1, pdf_text_2])

# Knowledge-based chatbot function
def knowledge_based_chatbot(user_query):
    relevant_docs = find_most_similar_documents(user_query, vectorizer, tfidf_matrix, [pdf_text_1, pdf_text_2])
    # Add OpenAI API call here

# User interaction loop
print("Chatbot: Hello! I'm a knowledge-based chatbot. How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = knowledge_based_chatbot(user_input)
    print("Chatbot:", response)
