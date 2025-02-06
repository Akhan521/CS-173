
import pandas as pd
import numpy as np
import spacy as sp
from collections import Counter

# A predefined Naive Bayes classifier to compare our implementation with.
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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

# A method to extract sentences from our dataframe, tokenize them, and return the tokens.
def preprocess_column(df, col_name):

    # Extracting our sentences from the given column.
    sents = df[col_name].dropna().tolist()

    # Tokenizing our sentences.
    tokens = []
    for sent in sents:
        # Tokenizing our sentence.
        sent_tokens = tokenize_sentence(sent)
        tokens += sent_tokens

    return tokens

# A method to preprocess our data and return the tokens for each emotion category.
def preprocess_data(df):

    # We'll extract our sadness data and preprocess it.
    sadness_tokens = preprocess_column(df, 'Sadness Sentences')

    # Our joy data.
    joy_tokens = preprocess_column(df, 'Joy Sentences')

    # Our fear data.
    fear_tokens = preprocess_column(df, 'Fear Sentences')

    # Our anger data.
    anger_tokens = preprocess_column(df, 'Anger Sentences')

    # Our surprise data.
    surprise_tokens = preprocess_column(df, 'Surprise Sentence')

    # Our disgust data.
    disgust_tokens = preprocess_column(df, 'Disgust Sentences')

    # Our sadness + joy data.
    sadness_joy_tokens = preprocess_column(df, 'Sadness + Joy Sentences')
    sadness_tokens += sadness_joy_tokens
    joy_tokens += sadness_joy_tokens

    # Our fear + anger data.
    fear_anger_tokens = preprocess_column(df, 'Fear + Anger Sentences')
    fear_tokens += fear_anger_tokens
    anger_tokens += fear_anger_tokens

    # Our surprise + disgust data.
    surprise_disgust_tokens = preprocess_column(df, 'Surprise + Disgust Sentences')
    surprise_tokens += surprise_disgust_tokens
    disgust_tokens += surprise_disgust_tokens

    # Our sadness + joy + fear data.
    sadness_joy_fear_tokens = preprocess_column(df, 'Sadness + Joy + Fear Sentences')
    sadness_tokens += sadness_joy_fear_tokens
    joy_tokens += sadness_joy_fear_tokens
    fear_tokens += sadness_joy_fear_tokens

    emotion_tokens = {'sadness': sadness_tokens, 'joy': joy_tokens, 'fear': fear_tokens,
                    'anger': anger_tokens, 'surprise': surprise_tokens, 'disgust': disgust_tokens}
    
    return emotion_tokens

# A method to calculate our prior probabilities, given our 6 emotion categories.
def calculate_priors(df):

    # Calculating the number of sentences for each emotion category.
    sadness_count = df['Sadness Sentences'].dropna().count()
    joy_count = df['Joy Sentences'].dropna().count()
    fear_count = df['Fear Sentences'].dropna().count()
    anger_count = df['Anger Sentences'].dropna().count()
    surprise_count = df['Surprise Sentence'].dropna().count()
    disgust_count = df['Disgust Sentences'].dropna().count()

    # Calculating the total number of sentences.
    total_count = sadness_count + joy_count + fear_count + anger_count + surprise_count + disgust_count

    # Calculating our prior probabilities in log space.
    sadness_prior = np.log(sadness_count / total_count)
    joy_prior = np.log(joy_count / total_count)
    fear_prior = np.log(fear_count / total_count)
    anger_prior = np.log(anger_count / total_count)
    surprise_prior = np.log(surprise_count / total_count)
    disgust_prior = np.log(disgust_count / total_count)

    priors = {'sadness': sadness_prior, 'joy': joy_prior, 'fear': fear_prior,
              'anger': anger_prior, 'surprise': surprise_prior, 'disgust': disgust_prior}
    
    return priors

# A method to calculate our likelihood probabilities, given our 6 emotion categories.
# We'll do everything in log space to avoid underflow and use add-one smoothing.
def calculate_likelihoods(emotion_tokens, vocab_size, sentence):

    # Our emotion categories.
    emotions = emotion_tokens.keys()

    # Initializing our likelihoods dictionary.
    likelihoods = {'sadness': 0, 'joy': 0, 'fear': 0, 'anger': 0, 'surprise': 0, 'disgust': 0}

    # For our convenience, we'll use counter objects for each emotion category.
    sadness_counter  = Counter(emotion_tokens['sadness'])
    joy_counter      = Counter(emotion_tokens['joy'])
    fear_counter     = Counter(emotion_tokens['fear'])
    anger_counter    = Counter(emotion_tokens['anger'])
    surprise_counter = Counter(emotion_tokens['surprise'])
    disgust_counter  = Counter(emotion_tokens['disgust'])

    # For each word in our sentence, we'll do the following:
    for word in sentence:

        # We'll compute P(word|emotion) for each emotion category using add-one smoothing.
        for emotion in emotions:

            # The number of times the word appears in the particular emotion category.
            numerator = 1
            if emotion == 'sadness':
                numerator += sadness_counter[word]
            elif emotion == 'joy':
                numerator += joy_counter[word]
            elif emotion == 'fear':
                numerator += fear_counter[word]
            elif emotion == 'anger':
                numerator += anger_counter[word]
            elif emotion == 'surprise':
                numerator += surprise_counter[word]
            elif emotion == 'disgust':
                numerator += disgust_counter[word]

            # The total number of words in the particular emotion category.
            denominator = len(emotion_tokens[emotion]) + vocab_size

            # Calculating the likelihood in log space.
            likelihoods[emotion] += np.log(numerator / denominator)

    return likelihoods    

# A method to predict the emotion category of a given sentence using a Naive Bayes classifier.
def predict_NB(emotion_tokens, sentence):

    # Tokenizing our test sentence.
    sentence = tokenize_sentence(sentence)

    # Calculating our prior probabilities.
    priors = calculate_priors(df)

    # Our vocabulary size.
    vocab = set()
    for emotion in emotion_tokens.keys():
        vocab.update(emotion_tokens[emotion])
    vocab_size = len(vocab)

    # Calculating our likelihood probabilities.
    likelihoods = calculate_likelihoods(emotion_tokens, vocab_size, sentence)

    # Calculating our posterior probabilities.
    # Recall that P(emotion|sentence) = P(sentence|emotion) * P(emotion). Also, we're working in log space.
    posteriors = {}

    for emotion in priors.keys():
        posteriors[emotion] = likelihoods[emotion] + priors[emotion]

    # Our predicted emotion category is the one with the highest posterior probability.
    # We'll return the predicted emotion and all the posterior probabilities.
    predicted_emotion = max(posteriors, key=posteriors.get)

    return predicted_emotion, posteriors


# Loading our data into a pandas dataframe.
df = pd.read_excel('data/CS173-published-sheet.xlsx')

# Preprocessing our data and extracting the tokens for each emotion category.
emotion_tokens = preprocess_data(df)

# Our test sentence.
sentence = '''
As she hugged her daughter goodbye on the first day of college, 
she felt both sad to see her go and joyfulknowing that she was 
embarking on a new and exciting chapter in her life.
'''

# Predicting the emotion category of our test sentence.
predicted_emotion, posteriors = predict_NB(emotion_tokens, sentence)

print(f'\nThe predicted emotion category for the given sentence is: {predicted_emotion}\n')

# Our test set will be the last 10 rows of our dataframe.
print('Evaluating our model on the test set...\n')
test_set = df.tail(10)

# We'll build a 6x6 confusion matrix to evaluate our model.
confusion_matrix = pd.DataFrame(0, index=emotion_tokens.keys(), columns=emotion_tokens.keys())

# Preprocessing our test set and extracting the tokens for each emotion category.
emotion_tokens = preprocess_data(test_set)

# Predicting the emotion category of each sentence in our test set.
for i, row in test_set.iterrows():
    sad_sent = row['Sadness Sentences']
    joy_sent = row['Joy Sentences']
    fear_sent = row['Fear Sentences']
    anger_sent = row['Anger Sentences']
    surprise_sent = row['Surprise Sentence']
    disgust_sent = row['Disgust Sentences']
    
    # Predicting the emotion category of our sadness sentence.
    predicted_emotion, _ = predict_NB(emotion_tokens, sad_sent)
    confusion_matrix.loc['sadness', predicted_emotion] += 1

    # Predicting the emotion category of our joy sentence.
    predicted_emotion, _ = predict_NB(emotion_tokens, joy_sent)
    confusion_matrix.loc['joy', predicted_emotion] += 1

    # Predicting the emotion category of our fear sentence.
    predicted_emotion, _ = predict_NB(emotion_tokens, fear_sent)
    confusion_matrix.loc['fear', predicted_emotion] += 1

    # Predicting the emotion category of our anger sentence.
    predicted_emotion, _ = predict_NB(emotion_tokens, anger_sent)
    confusion_matrix.loc['anger', predicted_emotion] += 1

    # Predicting the emotion category of our surprise sentence.
    predicted_emotion, _ = predict_NB(emotion_tokens, surprise_sent)
    confusion_matrix.loc['surprise', predicted_emotion] += 1

    # Predicting the emotion category of our disgust sentence.
    predicted_emotion, _ = predict_NB(emotion_tokens, disgust_sent)
    confusion_matrix.loc['disgust', predicted_emotion] += 1

print('Our confusion matrix is as follows:\n')
print(confusion_matrix)
print()

# Calculating the accuracy, precision, recall, and F1-score of our model on the Joy category of our test set.
print('Evaluating our model on the Joy category of the test set...\n')
TP = confusion_matrix.loc['joy', 'joy']
accuracy = TP / test_set.shape[0]
precision = TP / confusion_matrix['joy'].sum() # Summing across the row.
recall = TP / confusion_matrix[:]['joy'].sum() # Summing down the column.
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1_score * 100:.2f}%\n')

# We'll compare our implementation with a predefined Naive Bayes classifier.
df = df.head(40) # We'll use the first 40 rows of our dataframe as our training set.
x_train = []
y_train = []

# Using a CountVectorizer to convert our tokens into a matrix of token counts.
vectorizer = CountVectorizer()

# Iterating over our 40 rows and extracting the sentences.
for i, row in df.iterrows():
    
    if pd.notnull(row['Sadness Sentences']):
        tokens = tokenize_sentence(row['Sadness Sentences'])
        x_train.append(' '.join(tokens))
        y_train.append('sadness')
    
    if pd.notnull(row['Joy Sentences']):
        tokens = tokenize_sentence(row['Joy Sentences'])
        x_train.append(' '.join(tokens))
        y_train.append('joy')

    if pd.notnull(row['Fear Sentences']):
        tokens = tokenize_sentence(row['Fear Sentences'])
        x_train.append(' '.join(tokens))
        y_train.append('fear')

    if pd.notnull(row['Anger Sentences']):
        tokens = tokenize_sentence(row['Anger Sentences'])
        x_train.append(' '.join(tokens))
        y_train.append('anger')

    if pd.notnull(row['Surprise Sentence']):
        tokens = tokenize_sentence(row['Surprise Sentence'])
        x_train.append(' '.join(tokens))
        y_train.append('surprise')

    if pd.notnull(row['Disgust Sentences']):
        tokens = tokenize_sentence(row['Disgust Sentences'])
        x_train.append(' '.join(tokens))
        y_train.append('disgust')

x_train = vectorizer.fit_transform(x_train)

model = MultinomialNB()
model.fit(x_train, y_train)

# Our test set will be the last 10 rows of our dataframe.
print('Evaluating the predefined Naive Bayes classifier on the test set...\n')
test_set = df.tail(10)
x_test = []
y_test = []

# Iterating over our 10 rows and extracting the sentences.
for i, row in test_set.iterrows():
    
    if pd.notnull(row['Sadness Sentences']):
        tokens = tokenize_sentence(row['Sadness Sentences'])
        x_test.append(' '.join(tokens))
        y_test.append('sadness')
    
    if pd.notnull(row['Joy Sentences']):
        tokens = tokenize_sentence(row['Joy Sentences'])
        x_test.append(' '.join(tokens))
        y_test.append('joy')

    if pd.notnull(row['Fear Sentences']):
        tokens = tokenize_sentence(row['Fear Sentences'])
        x_test.append(' '.join(tokens))
        y_test.append('fear')

    if pd.notnull(row['Anger Sentences']):
        tokens = tokenize_sentence(row['Anger Sentences'])
        x_test.append(' '.join(tokens))
        y_test.append('anger')

    if pd.notnull(row['Surprise Sentence']):
        tokens = tokenize_sentence(row['Surprise Sentence'])
        x_test.append(' '.join(tokens))
        y_test.append('surprise')

    if pd.notnull(row['Disgust Sentences']):
        tokens = tokenize_sentence(row['Disgust Sentences'])
        x_test.append(' '.join(tokens))
        y_test.append('disgust')

x_test = vectorizer.transform(x_test)
predictions = model.predict(x_test)

# We'll build a 6x6 confusion matrix to evaluate our model.
confusion_matrix = pd.DataFrame(0, index=model.classes_, columns=model.classes_)
for i in range(len(predictions)):
    confusion_matrix.loc[y_test[i], predictions[i]] += 1

print('Our confusion matrix is as follows:\n')
print(confusion_matrix)
print()







