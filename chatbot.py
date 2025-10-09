import random
import json
import pickle
import numpy as np

import tkinter as tk
from tkinter import scrolledtext

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.keras")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words   

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json["intents"]
    result=""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result
    
    
# --- Interfaz gráfica ---
def send_message():
    msg = user_input.get()
    if msg.strip() != "":
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, "You: " + msg + "\n", "user")
        user_input.delete(0, tk.END)

        ints = predict_class(msg)
        res = get_response(ints, intents)

        chat_box.insert(tk.END, "Aisha: " + res + "\n\n", "bot")
        chat_box.config(state=tk.DISABLED)
        chat_box.yview(tk.END)

# Crear ventana
root = tk.Tk()
root.title("Aisha Chatbot 💬")
root.geometry("400x500")
root.resizable(False, False)

# Caja de chat
chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Entrada de usuario
user_input = tk.Entry(root, font=("Arial", 12))
user_input.pack(padx=10, pady=5, fill=tk.X)
user_input.bind("<Return>", lambda event: send_message())

# Botón de enviar
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

# Colores opcionales
chat_box.tag_config("user", foreground="blue")
chat_box.tag_config("bot", foreground="green")

root.mainloop()
    
    
