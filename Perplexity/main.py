import spacy
from sklearn.model_selection import train_test_split
import re

# A function to generate n-grams from a list of tokens.
def create_n_grams(tokens, n):
    size = len(tokens)
    n_grams = []

    # We need to have enough tokens to begin with...
    if n > size:
        return n_grams

    for i in range(size - n + 1):     # Leaving enough space at the end for the final n_gram.
        n_grams.append(tokens[i:i+n]) # Reading n tokens at a time.
    return n_grams

# A function to count the occurrences of each n-gram in a list of n-grams.
def count_n_grams(n_grams):
    n_gram_freqs = {}
    for n_gram in n_grams:
        if n_gram in n_gram_freqs:
            n_gram_freqs[n_gram] += 1
        else:
            n_gram_freqs[n_gram] = 1
    return n_gram_freqs

# A function to calculate the bigram probability given list of n-grams.
# Our formula is: P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
# We also provide our tokens to calculate the count of w_{i-1}.
# Using add-1 smoothing to avoid zero probabilities.
def calculate_bigram_prob(tokens, n_grams):
    # Storing the counts of each bigram.
    bigram_counts = count_n_grams(n_grams)
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        # A count of w_{i-1}.
        count_w_i_1 = tokens.count(bigram[0])
        # Calculating the probability using add-1 smoothing.
        # P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + |V|)
        bigram_probs[bigram] = (count + 1) / (count_w_i_1 + len(tokens))

    return bigram_probs

# A function to generate text using a bigram model given our test set and bigram probabilities.
def generate_text(test_set, n, size): # Here, n = 2, and size is the number of tokens we want to generate.

    generated_text = []
    # Getting our bigrams from the test set.
    test_bigrams = create_n_grams(test_set, n)
    # Getting our bigram probabilities.
    bigram_probs = calculate_bigram_prob(test_set, test_bigrams)
    # Starting with the first token in the test set. Thereafter, we will generate the next token using the bigram model.
    generated_text.append(test_set[0])
    for i in range(size - 1):
        # Our bigram model approach:
        max_prob = 0.0
        next_token = None
        for token in test_set:
            # We need to find the token that follows the last token in our generated text.
            bigram = [generated_text[-1], token]
            # If the bigram exists in our bigram probabilities, we will use it.
            if bigram in bigram_probs:
                # We will choose the token with the highest probability.
                if bigram_probs[bigram] > max_prob:
                    max_prob = bigram_probs[bigram]
                    next_token = token
        generated_text.append(next_token)

    return generated_text
    


nlp = spacy.load('en_core_web_sm')
file = open("data/cs173_nlp_movie.txt", "rb")
text = file.read()
text = text.decode("utf-8")

# Lowercase the text
text = text.lower()

# Tokenizing the text
doc = nlp(text)
tokens = [token for token in doc]

# Handling punctuation and special characters
tokens = [token for token in tokens if token.is_alpha or token.is_digit]
print(tokens)

# Splitting our text into 3 sets: training, validation, and test.
train, test = train_test_split(tokens, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

# Obtaining our n-grams for n=2.
n = 2
training_n_grams = create_n_grams(train, n)

# Counting the occurrences of each n-gram in the training set.
training_n_gram_freqs = count_n_grams(training_n_grams)

# Calculating the bigram probabilities.
bigram_probs = calculate_bigram_prob(train, training_n_grams)

