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

        # --- Mensaje del usuario (alineado a la derecha) ---
        chat_box.insert(tk.END, "\n", "right_space")
        chat_box.insert(tk.END, f" Tú: {msg} ", "user_bubble")
        chat_box.insert(tk.END, "\n", "right_space")
        user_input.delete(0, tk.END)

        # --- Procesar mensaje con tu chatbot ---
        ints = predict_class(msg)
        res = get_response(ints, intents)

        # --- Mensaje del bot (alineado a la izquierda) ---
        chat_box.insert(tk.END, "\n", "left_space")
        chat_box.insert(tk.END, f" Aisha: {res} ", "bot_bubble")
        chat_box.insert(tk.END, "\n", "left_space")

        chat_box.config(state=tk.DISABLED)
        chat_box.yview(tk.END)

# --- Ventana principal ---
root = tk.Tk()
root.title("Aisha Chatbot 💬")
root.geometry("420x550")
root.resizable(False, False)
root.configure(bg="#e5ddd5")  # Color tipo WhatsApp

# --- Caja de chat ---
chat_box = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, state=tk.DISABLED,
    font=("Segoe UI", 10), bg="#ece5dd",
    relief=tk.FLAT, bd=0, padx=10, pady=10
)
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# --- Estilos de burbujas ---
chat_box.tag_config("user_bubble", 
                    background="#cce5ff",  # Azul clarito (usuario)
                    foreground="#000000", 
                    justify="right",
                    spacing3=10,
                    lmargin1=150, rmargin=10,
                    font=("Segoe UI", 10))

chat_box.tag_config("bot_bubble", 
                    background="#e1f3fb",  # Azul suave (Aisha)
                    foreground="#000000", 
                    justify="left",
                    spacing3=10,
                    lmargin1=10, rmargin=150,
                    font=("Segoe UI", 10))

chat_box.tag_config("left_space", justify="left")
chat_box.tag_config("right_space", justify="right")

# --- Entrada de usuario ---
bottom_frame = tk.Frame(root, bg="#f0f0f0")
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

user_input = tk.Entry(bottom_frame, font=("Segoe UI", 11),
                      bd=0, relief=tk.FLAT, bg="white")
user_input.pack(side=tk.LEFT, padx=(10, 0), pady=10, fill=tk.X, expand=True)
user_input.bind("<Return>", lambda event: send_message())

# --- Botón de enviar ---
send_button = tk.Button(bottom_frame, text="Enviar 💬",
                        command=send_message, bg="#128C7E", fg="white",
                        font=("Segoe UI", 10, "bold"), relief=tk.FLAT,
                        padx=15, pady=5)
send_button.pack(side=tk.RIGHT, padx=(5, 10), pady=10)

root.mainloop()
    
