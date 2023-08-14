import openai
import os
from config import OPENAI_API_KEY
from pdf_extraction import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Load PDF content (replace with your PDF paths)
pdf_path_1 = "document1.pdf"
pdf_path_2 = "document2.pdf"

pdf_text_1 = extract_text_from_pdf(pdf_path_1)
pdf_text_2 = extract_text_from_pdf(pdf_path_2)

# Preprocess and index PDF texts
def preprocess_and_index(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = preprocess_and_index([pdf_text_1, pdf_text_2])

# Find most similar documents based on user query
def find_most_similar_documents(query, vectorizer, tfidf_matrix, texts):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    most_similar_idx = similarity_scores.argsort()[-5:][::-1]
    return [texts[idx] for idx in most_similar_idx]

# Knowledge-based chatbot function
def knowledge_based_chatbot(user_query):
    relevant_docs = find_most_similar_documents(user_query, vectorizer, tfidf_matrix, [pdf_text_1, pdf_text_2])
    
    chatbot_response = f"I found relevant information in the documents:\n"
    for idx, doc in enumerate(relevant_docs, 1):
        chatbot_response += f"Document {idx}:\n{doc}\n\n"
    
    openai_response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=chatbot_response + f"User: {user_query}\nChatbot:",
        max_tokens=150
    )

    return openai_response.choices[0].text.strip()

# User interaction loop
print("Chatbot: Hello! I'm a knowledge-based chatbot. How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = knowledge_based_chatbot(user_input)
    print("Chatbot:", response)
