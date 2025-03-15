import pandas as pd
import numpy as np
import spacy as sp
from collections import Counter

# Loading the spacy model.
nlp = sp.load('en_core_web_sm')

# Loading our stopwords.
stopwords = sp.lang.en.stop_words.STOP_WORDS

# A method to tokenize a given sentence and return the tokens.
def tokenize_sentence(sentence):
    
    doc = nlp(sentence)
    tokens = [token.text.lower() for token in doc]
    tokens = [token for token in tokens if token.isalpha() and token not in stopwords]

    return tokens

# A method to filter our data based on the given columns.
# We provide our dataframe and the columns we'd like to extract data from.
def filter_data(df, cols):

    # We create a new dataframe to store our filtered data.
    filtered_df = pd.DataFrame()

    for col in cols:
        filtered_df[col] = df[col]

    return filtered_df

# A method to extract our lexicons from the given data.
def extract_lexicons():

    # Loading the NRC Emotion Lexicons.
    filepath = 'data/nrc_emotion_lexicons.txt'
    emolex = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)

    # We'll filter the lexicons to only contain "Joy" and "Sadness" labels.
    filtered_emolex = emolex[emolex['emotion'].isin(['joy', 'sadness'])]

    # Moreover, we'll filter the lexicons to only contain words with a positive association.
    filtered_emolex = filtered_emolex[filtered_emolex['association'] == 1]

    # We'll create a dictionary to store our lexicons.
    lexicons = {'Joy Lexicons': [], 'Sadness Lexicons': []}

    # We'll iterate through our filtered lexicons and store them in our dictionary.
    for index, row in filtered_emolex.iterrows():
        if row['emotion'] == 'joy':
            lexicons['Joy Lexicons'].append(row['word'])
        else:
            lexicons['Sadness Lexicons'].append(row['word'])

    return lexicons

# A method to get the tf-idf score for the given token.
def get_tfidf(token, doc, docs):

    counter = Counter(doc)

    # Calculating the term frequency.
    tf = counter[token] / len(doc)

    # Calculating the inverse document frequency.
    idf = np.log(len(docs) / 1 + sum([1 for doc in docs if token in doc]))

    return tf * idf

# A method to vectorize an input sentence based on the given lexicons.
# x1 = Joy Lexicons, x2 = Sadness Lexicons, x3 = tf-idf score for Joy Lexicons, x4 = tf-idf score for Sadness Lexicons, x5 = Total Words.
def vectorize_sentence(sentence, lexicons):

    # Tokenizing the input sentence.
    tokens = tokenize_sentence(sentence)

    # Initializing our vector.
    vector = np.zeros(5)

    for token in tokens:
        if token in lexicons['Joy Lexicons']:
            vector[0] += 1
        if token in lexicons['Sadness Lexicons']:
            vector[1] += 1
        vector[2] += get_tfidf(token, tokens, lexicons['Joy Lexicons'])
        vector[3] += get_tfidf(token, tokens, lexicons['Sadness Lexicons'])
    vector[4] = len(tokens)

    return vector

# Our sigmoid activation function.
def sigmoid(z):
    # To prevent overflow, we'll clip z to be within [-500, 500].
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Our cross-entropy loss function.
def loss(y, y_hat):
    # Defining a small constant to prevent log(0).
    epsilon = 1e-10
    # Ensuring y_hat is within [epsilon, 1 - epsilon].
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Our logistic regression model.
# P( y = 1 | x ) = sigmoid( z ) = sigmoid( w * x + b ), where 1 = Joy, 0 = Sadness.
# x = [x1, x2, x3] = [Joy Lexicons, Sadness Lexicons, Total Words]
# w = [w1, w2, w3]
def log_regr_classifier(w, x, b):

    z = np.dot(w, x) + b
    y_hat = sigmoid(z)
    return y_hat

# Our Stochastic Gradient Descent (SGD) method.
def sgd(X_train, y_train, X_val, y_val, epochs, lr):

    # Setting up our parameters.
    w, b = np.zeros(5), 0

    # To store our validation loss history.
    loss_history = []

    for epoch in range(epochs):

        # Selecting a random training example.
        idx = np.random.randint(0, len(X_train) - 1)
        x, y = X_train[idx], y_train[idx]

        # Computing our prediction.
        y_hat = log_regr_classifier(w, x, b)

        # Calculating our gradients.
        dw = (y_hat - y) * x
        db = y_hat - y

        # Updating our parameters.
        w -= lr * dw
        b -= lr * db

        # Calculating our training loss on the validation set.
        val_loss = 0
        for i in range(len(X_val)):
            x, y = X_val[i], y_val[i]
            y_hat = log_regr_classifier(w, x, b)
            val_loss += loss(y, y_hat)
        val_loss /= len(X_val)
        loss_history.append(val_loss)

    return w, b, loss_history


# Our Program:
# --------------------------------------------------------------------------------------------------------------------

# Filtering our data to only contain 'Joy Sentences' and 'Sadness Sentences'.
df = pd.read_excel('data/CS173-published-sheet.xlsx')
df = filter_data(df, ['Joy Sentences', 'Sadness Sentences', 'Sadness + Joy Sentences'])
training   = df.head(30) # Using the first 30 rows as our training data.
validation = df[30:40]   # Using the next 10 rows as our validation data.
testing    = df.tail(10) # Using the last 10 rows as our testing data.

# Extracting our lexicons.
lexicons = extract_lexicons()

# Vectorizing our training data.
X_train, y_train = [], []

for index, row in training.iterrows():
    joy_sentence = row['Joy Sentences']
    sad_sentence = row['Sadness Sentences']
    joy_vector = vectorize_sentence(joy_sentence, lexicons)
    sad_vector = vectorize_sentence(sad_sentence, lexicons)
    X_train.append(joy_vector)
    X_train.append(sad_vector)
    y_train.append(1) # Joy
    y_train.append(0) # Sadness

X_train, y_train = np.array(X_train), np.array(y_train)

# Vectorizing our validation data.
X_val, y_val = [], []

for index, row in validation.iterrows():
    joy_sentence = row['Joy Sentences']
    sad_sentence = row['Sadness Sentences']
    joy_vector = vectorize_sentence(joy_sentence, lexicons)
    sad_vector = vectorize_sentence(sad_sentence, lexicons)
    X_val.append(joy_vector)
    X_val.append(sad_vector)
    y_val.append(1) # Joy
    y_val.append(0) # Sadness

X_val, y_val = np.array(X_val), np.array(y_val)

# Training our model using Stochastic Gradient Descent.
epochs = 1000
best_rate = None         # To store the best learning rate.
best_loss = float('inf') # To store the best validation loss.
best_w = None            # To store the best weights.
best_b = None            # To store the best bias.

# Testing different learning rates.
rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

for lr in rates:
    
    w, b, loss_history = sgd(X_train, y_train, X_val, y_val, epochs, lr)
    val_loss = loss_history[-1]

    # print(f'Learning Rate: {lr} - Validation Loss: {val_loss:.4f}')
    # print(f'------------------------------------------------')

    if val_loss < best_loss:
        best_rate = lr
        best_loss = val_loss
        best_w = w
        best_b = b

# print(f'Best Learning Rate: {best_rate}')
# print(f'Best Validation Loss: {best_loss:.4f}')

# Vectorizing our testing data.
X_test, y_test = [], []

for index, row in testing.iterrows():
    joy_sentence = row['Joy Sentences']
    sad_sentence = row['Sadness Sentences']
    joy_vector = vectorize_sentence(joy_sentence, lexicons)
    sad_vector = vectorize_sentence(sad_sentence, lexicons)
    X_test.append(joy_vector)
    X_test.append(sad_vector)
    y_test.append(1) # Joy
    y_test.append(0) # Sadness

X_test, y_test = np.array(X_test), np.array(y_test)

# Testing our model on the testing data and generating our confusion matrix.
conf_mat = np.zeros((2, 2))
for (x, y) in zip(X_test, y_test):
    y_hat = log_regr_classifier(best_w, x, best_b)
    y_hat = 1 if y_hat >= 0.5 else 0
    conf_mat[y, y_hat] += 1

# Outputting our confusion matrix as a DataFrame.
conf_mat = pd.DataFrame(conf_mat, columns=['Sadness', 'Joy'], index=['Sadness', 'Joy'])
print(f'Confusion Matrix:')
print(f'------------------------')
print(conf_mat)
print(f'------------------------')
print(f"w  = {best_w}")
print(f"b  = {best_b}")
print(f"lr = {best_rate}")
print(f"loss = {best_loss}")

# Calculating our accuracy for the Joy category.
joy_accuracy = conf_mat.iloc[1, 1] / (conf_mat.iloc[1, 0] + conf_mat.iloc[1, 1])
print(f'Joy Accuracy: {joy_accuracy*100:.2f}%')

# Calculating our recall for the Joy category.
joy_recall = conf_mat.iloc[1, 1] / (conf_mat.iloc[0, 1] + conf_mat.iloc[1, 1])
print(f'Joy Recall: {joy_recall*100:.2f}%')

# Calculating our precision for the Joy category.
joy_precision = conf_mat.iloc[1, 1] / (conf_mat.iloc[1, 0] + conf_mat.iloc[1, 1])
print(f'Joy Precision: {joy_precision*100:.2f}%')

# Calculating our F1 Score for the Joy category.
joy_f1 = 2 * (joy_precision * joy_recall) / (joy_precision + joy_recall)
print(f'Joy F1 Score: {joy_f1*100:.2f}%')


