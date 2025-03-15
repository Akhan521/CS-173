import spacy
import math
import numpy as np

nlp = spacy.load('en_core_web_sm')
file = open("data/cs173_nlp_movie.txt", "rb")
text = file.read()
text = text.decode("utf-8")

# Lowercase the text
text = text.lower()

# Tokenize the text
doc = nlp(text)
tokens = [token.text for token in doc]

# Our vocabulary
vocab = list(set(tokens))

# A method to compute the term frequency.
def tf(term, tokenized_document):
    return tokenized_document.count(term) / len(tokenized_document)

# A method to compute the inverse document frequency.
def idf(term, all_tokenized_documents):
    total_docs = len(all_tokenized_documents)
    docs_with_term = len([doc for doc in all_tokenized_documents if term in doc])
    return math.log(total_docs / 1 + docs_with_term)

# A method to compute the TF-IDF vector for a given document (set of terms).
def vectorize(doc, vocab, all_tokenized_documents):
    vector = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        vector[i] = tf(term, doc) * idf(term, all_tokenized_documents)
    return vector

# Compute the cosine similarity of two vectors.
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Going through an example.
all_docs = [tokens]
vector = vectorize(tokens, vocab, all_docs)
print(vector)

# Compute the cosine similarity between two documents.
doc1 = "I like apples and oranges"
doc2 = "I like apples and bananas"
doc3 = "I like cars and trucks"
doc1_tokens = [token.text for token in nlp(doc1)]
doc2_tokens = [token.text for token in nlp(doc2)]
doc3_tokens = [token.text for token in nlp(doc3)]
all_docs = [doc1_tokens, doc2_tokens, doc3_tokens]
vocab = list(set(doc1_tokens + doc2_tokens + doc3_tokens))
doc1_vector = vectorize(doc1_tokens, vocab, all_docs)
doc2_vector = vectorize(doc2_tokens, vocab, all_docs)
doc3_vector = vectorize(doc3_tokens, vocab, all_docs)
sim_1_2 = cos_sim(doc1_vector, doc2_vector)
sim_1_3 = cos_sim(doc1_vector, doc3_vector)
sim_2_3 = cos_sim(doc2_vector, doc3_vector)

print(f"\nDocument 1: {doc1}")
print(f"Document 2: {doc2}")
print(f"Document 3: {doc3}")

print(f"\nSimilarity between Document 1 and Document 2: {sim_1_2}")
print(f"Similarity between Document 1 and Document 3: {sim_1_3}")
print(f"Similarity between Document 2 and Document 3: {sim_2_3}")
