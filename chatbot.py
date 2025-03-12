import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

file_path = os.path.abspath("intents.json")
with open(file_path,"r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

X = vectorizer.fit_transform(patterns)
Y = tags
clf.fit(X,Y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent ['tag'] == tag :
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter 
    st.title("Intents of Chatbot using NLP")

    menu = ["Home","Conversation History","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == 'Home':
        st.write("Welcome to the Chatbot. Please ask your queries and Press enter after that!")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv','w',newline='',encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input','Chatbot Response','Timestamp'])
        
        counter +=1
        user_input = st.text_input("You",key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=100, max_chars=None,
                         key=f"chatbot_response_{counter}")
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M;%S")

            with open ('chat_log.csv', 'a', newline='',encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])
            
            if response.lower() in ['goodbye', 'bye', 'take care']:
                st.write("Thank you for the chatting with me, have a good day!")
                st.stop()

    elif choice == 'Conversation History':
        st.header("Conversation History")
        with open('chat_log.csv', 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown('---')

    elif choice == 'About':
        st.write("The goal of this project is to showcase that we create chatbot using python with NLP")
        
        st.subheader("Project Overview")
        st.write('''
        The project is divided into the following parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labled intents.
        2.To build the interface, we used Streamlit as an interactive app''')

        st.subheader("Chatbot Interface:")
        st.write("The chatbot interface is simple,it takes user input and returns the reponse from the chatbot after pressing  the enter key.")
         
        st.subheader("conclusion")
        st.write('We build the chatbot using NLP and LR, and using streamlit for the interactive app development,This project can be further more improved with more data,sophisticated NLP techniques, lastly we can use Deep learning')

if __name__ == '__main__':
    main()






