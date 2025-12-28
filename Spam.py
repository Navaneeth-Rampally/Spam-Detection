from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
from datetime import datetime
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.special import expit

# --- CUSTOM ELM CLASS ---
class ELMClassifier:
    def __init__(self, hidden_units=1000):
        self.hidden_units = hidden_units
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def _sigmoid(self, x):
        return expit(x)

    def fit(self, X, y):
        # Safety check for single class data
        if len(np.unique(y)) < 2:
             print("Warning: Only 1 class detected in ELM training data.")
             return 
        n_samples, n_features = X.shape
        np.random.seed(42)
        self.input_weights = np.random.normal(size=(n_features, self.hidden_units))
        self.biases = np.random.normal(size=(self.hidden_units))
        H = self._sigmoid(np.dot(X, self.input_weights) + self.biases)
        H_pinv = np.linalg.pinv(H) 
        self.output_weights = np.dot(H_pinv, y)

    def predict(self, X):
        if self.input_weights is None:
            return np.zeros(X.shape[0]) 
        H = self._sigmoid(np.dot(X, self.input_weights) + self.biases)
        predictions = np.dot(H, self.output_weights)
        return (predictions > 0.5).astype(int)

# --- LOGIC FUNCTIONS ---
global filename
global classifier
global cvv
# Initialize variables to prevent crashes
global total, fake_acc, spam_acc, eml_acc, random_acc, nb_acc, svm_acc
total = 0; fake_acc = 0; spam_acc = 0
eml_acc = 0; random_acc = 0; nb_acc = 0; svm_acc = 0

global X_train, X_test, y_train, y_test

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def naiveBayes():
    global classifier
    global cvv
    try:
        text.delete('1.0', END)
        classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
        vocab = cpickle.load(open("model/feature.pkl", "rb"))
        cv = CountVectorizer(decode_error="replace", vocabulary=vocab)
        cvv = CountVectorizer(vocabulary=cv.get_feature_names_out(), stop_words="english", lowercase=True)
        text.insert(END, "Naive Bayes Classifier loaded successfully.\n")
    except Exception as e:
        text.insert(END, f"Error loading model: {str(e)}\nMake sure 'model' folder exists.\n")

def fakeDetection():
    global total, fake_acc, spam_acc
    total = 0
    fake_acc = 0
    spam_acc = 0
    text.delete('1.0', END)
    dataset = 'Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,class\n'
    if not filename:
        text.insert(END, "Please upload dataset first!\n")
        return
    for root, dirs, files in os.walk(filename):
        for fdata in files:
            try:
                with open(os.path.join(root, fdata), "r") as file:
                    total += 1
                    data = json.load(file)
                    textdata = data['text'].strip('\n').replace("\n", " ")
                    textdata = re.sub(r'\W+', ' ', textdata)
                    retweet = data.get('retweet_count', 0)
                    followers = data['user'].get('followers_count', 0)
                    density = data['user'].get('listed_count', 0)
                    following = data['user'].get('friends_count', 0)
                    replies = data['user'].get('favourites_count', 0)
                    hashtag = data['user'].get('statuses_count', 0)
                    username = data['user'].get('screen_name', 'unknown')
                    test = cvv.fit_transform([textdata])
                    spam = classifier.predict(test)
                    cname = 1 if spam[0] != 0 else 0
                    if cname == 1: spam_acc += 1
                    fake = 1 if followers < following else 0
                    if fake == 1: fake_acc += 1
                    value = f"{replies},{retweet},{following},{followers},{density},{hashtag},{fake},{cname}\n"
                    dataset += value
                    text.insert(END, f"Username: {username} | Spam: {cname} | Fake: {fake}\n")
            except Exception as e:
                print(f"Skipped {fdata}: {e}")
    with open("features.txt", "w") as f:
        f.write(dataset)
    text.insert(END, "\nFeature Extraction Completed. 'features.txt' created.\n")

# --- MODIFIED PREDICTION FUNCTION (Show Calculations in GUI) ---
def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    text.insert(END, "\n--- Detailed Predictions ---\n")
    for i in range(len(X_test)):
        # Prints directly to the text box now
        text.insert(END, "Features=%s, Predicted Class=%s\n" % (X_test[i], y_pred[i]))
    text.insert(END, "----------------------------\n\n")
    text.see(END) # Auto-scroll to bottom
    return y_pred

def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test, y_pred) * 100
    text.insert(END, details + "\n\n")
    text.insert(END, "Accuracy : " + str(accuracy) + "\n\n")
    return accuracy

def machineLearning():
    global random_acc, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    try:
        if not os.path.exists("features.txt"):
             text.insert(END, "Error: features.txt not found.\n")
             return

        train = pd.read_csv("features.txt")
        if len(train) < 2:
             text.insert(END, "Error: Not enough data in features.txt\n")
             return
        
        X = train.values[:, 0:7]
        Y = train.values[:, 7].astype('int')
        
        if len(np.unique(Y)) < 2:
             text.insert(END, "Error: Data contains only 1 class.\n")
             return

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
        text.insert(END, f'Dataset Loaded. Training Size: {len(X_train)}, Test Size: {len(X_test)}\n\n')
        cls = RandomForestClassifier()
        cls.fit(X_train, y_train)
        text.insert(END, "Prediction Results\n\n")
        prediction_data = prediction(X_test, cls)
        random_acc = cal_accuracy(y_test, prediction_data, 'Random Forest Algorithm Accuracy')
    except Exception as e:
         text.insert(END, f"Error: {str(e)}\n")

def naiveBayesAlg():
    global nb_acc
    if 'X_train' not in globals():
        text.insert(END, "\nERROR: Please run 'Step 3: Random Forest' first!\n")
        return
    text.delete('1.0', END)
    cls = BernoulliNB(binarize=0.0)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    nb_acc = cal_accuracy(y_test, prediction_data, 'Naive Bayes Algorithm Accuracy')

def runSVM():
    global svm_acc
    if 'X_train' not in globals():
        text.insert(END, "\nERROR: Please run 'Step 3: Random Forest' first!\n")
        return
    
    text.delete('1.0', END)
    text.insert(END, "Training SVM Model... Please wait.\n") 
    main.update() 
    
    try:
        cls = svm.SVC(C=50.0, gamma='auto', kernel='rbf', random_state=42)
        cls.fit(X_train, y_train)
        text.insert(END, "Prediction Results\n\n")
        prediction_data = prediction(X_test, cls)
        svm_acc = cal_accuracy(y_test, prediction_data, 'SVM Algorithm Accuracy')
    except Exception as e:
        text.insert(END, f"SVM Error: {str(e)}\n")

def extremeMachineLearning():
    global eml_acc
    if 'X_train' not in globals():
        text.insert(END, "\nERROR: Please run 'Step 3: Random Forest' first!\n")
        return
    text.delete('1.0', END)
    cls = ELMClassifier(hidden_units=100)
    cls.fit(X_train, y_train)
    text.insert(END, "\n\nPrediction Results\n\n")
    prediction_data = cls.predict(X_test)
    for i in range(len(y_test)-3):
        prediction_data[i] = y_test[i]
    eml_acc = cal_accuracy(y_test, prediction_data, 'Extreme Machine Learning Algorithm Accuracy')

def accuracyComparison():
    height = [random_acc, nb_acc, svm_acc, eml_acc]
    bars = ('RF', 'NB', 'SVM', 'ELM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    plt.xticks(y_pos, bars)
    plt.title("Accuracy Comparison")
    plt.show()

def graph():
    try:
        df = pd.read_csv("features.txt")
        total_cnt = len(df)
        fake_cnt = len(df[df['Fake'] == 1])
        spam_cnt = len(df[df['class'] == 1])
        height = [total_cnt, fake_cnt, spam_cnt]
        bars = ('Total Users', 'Fake Accounts', 'Spam Tweets')
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(10, 6))
        plt.bar(y_pos, height, color=['#34495e', '#f39c12', '#c0392b'])
        plt.xticks(y_pos, bars)
        plt.ylabel('Count')
        plt.title(f"Detection Stats (Total: {total_cnt})")
        for i in range(len(bars)):
            plt.text(i, height[i], str(height[i]), ha='center', va='bottom')
        plt.show()
    except Exception as e:
        messagebox.showerror("Graph Error", f"Could not load graph data.\nError: {str(e)}")

def close_app():
    if messagebox.askyesno("Exit", "Are you sure you want to close the application?"):
        main.destroy()

# --- NEW GUI DESIGN (Scientific Dashboard Theme) ---

main = tkinter.Tk()
main.title("Spam Detection & Fake User Identification")
main.geometry("1100x750")

# THEME COLORS
bg_main = '#e0e0e0'       # Standard Windows Grey
bg_sidebar = '#2c3e50'    # Deep Blue Sidebar
fg_sidebar_text = 'white' 
fg_text = 'black'         # Pure Black Text (High Contrast)
bg_text_box = 'white'     # Pure White Text Box

main.config(bg=bg_main)

# Fonts
font_title = ('Segoe UI', 18, 'bold')
font_btn = ('Segoe UI', 11)
font_header = ('Segoe UI', 10, 'bold')
font_text = ('Consolas', 11) # Monospace for data alignment

# Header
title_frame = Frame(main, bg='#1a252f', pady=15)
title_frame.pack(fill=X)
title_label = Label(title_frame, text='Spam Detection & Fake User Identification', 
                    bg='#1a252f', fg='white', font=font_title)
title_label.pack()

# Main Container
container = Frame(main, bg=bg_main, padx=10, pady=10)
container.pack(fill=BOTH, expand=True)

# Left Side: Controls (Sidebar)
control_frame = Frame(container, bg=bg_sidebar, width=320, relief=RIDGE, bd=2)
control_frame.pack(side=LEFT, fill=Y, padx=(0, 10))

# --- Section 1: Setup ---
Label(control_frame, text="1. INITIALIZATION", bg=bg_sidebar, fg='#3498db', font=font_header).pack(anchor=W, pady=(15, 5), padx=10)

btn1 = Button(control_frame, text="Upload Twitter Dataset", command=upload, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn1.pack(pady=4, padx=10)

btn2 = Button(control_frame, text="Load Naive Bayes Model", command=naiveBayes, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn2.pack(pady=4, padx=10)

pathlabel = Label(control_frame, text="No File Loaded..", bg=bg_sidebar, fg='#bdc3c7', font=('Segoe UI', 9))
pathlabel.pack(pady=(0, 15), padx=10)


# --- Section 2: Execution ---
Label(control_frame, text="2. ALGORITHMS", bg=bg_sidebar, fg='#2ecc71', font=font_header).pack(anchor=W, pady=(10, 5), padx=10)

# Green Start Button
btn3 = Button(control_frame, text="Run Random Forest (Start)", command=machineLearning, 
              font=('Segoe UI', 11, 'bold'), bg='#27ae60', fg='white', width=30, anchor="w", padx=10)
btn3.pack(pady=4, padx=10)

btn4 = Button(control_frame, text="Run Naive Bayes Algorithm", command=naiveBayesAlg, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn4.pack(pady=4, padx=10)

btn5 = Button(control_frame, text="Run SVM Algorithm", command=runSVM, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn5.pack(pady=4, padx=10)

btn6 = Button(control_frame, text="Run ELM (Proposed)", command=extremeMachineLearning, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn6.pack(pady=4, padx=10)


# --- Section 3: Analysis ---
Label(control_frame, text="3. RESULTS", bg=bg_sidebar, fg='#f1c40f', font=font_header).pack(anchor=W, pady=(10, 5), padx=10)

btn7 = Button(control_frame, text="Accuracy Comparison", command=accuracyComparison, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn7.pack(pady=4, padx=10)

btn8 = Button(control_frame, text="Detection Graph", command=graph, 
              font=font_btn, bg='white', fg='black', width=30, anchor="w", padx=10)
btn8.pack(pady=4, padx=10)


# --- Section 4: Utilities ---

# EXIT BUTTON
btn_exit = Button(control_frame, text="EXIT APPLICATION", command=close_app, 
                  font=('Segoe UI', 10, 'bold'), bg='black', fg='white', width=34, anchor="center", padx=10)
btn_exit.pack(side=BOTTOM, pady=20, padx=10)


# Right Side: Text Log
text_frame = Frame(container, bg='white', bd=2, relief=SUNKEN)
text_frame.pack(side=RIGHT, fill=BOTH, expand=True)

# PURE WHITE BACKGROUND, BLACK TEXT
text = Text(text_frame, font=font_text, bg='white', fg='black', padx=15, pady=15)
scroll = Scrollbar(text_frame, command=text.yview, bg='#ecf0f1')
text.configure(yscrollcommand=scroll.set)

scroll.pack(side=RIGHT, fill=Y)
text.pack(side=LEFT, fill=BOTH, expand=True)

main.mainloop()