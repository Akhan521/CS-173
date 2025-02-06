# from collections import Counter
# #import ipdb
# def generate_merges(corpus, num_merges):
#     vocab = Counter([" ".join(list(word)) + " _" for word in corpus])
#     print(vocab)
#     merges = []
#     for _ in range(num_merges):
#         pairs = Counter()
#         for word, freq in vocab.items():
#             tokens = word.split()
#             print(tokens)
#             for i in range(len(tokens) - 1):
#                 pairs[(tokens[i], tokens[i + 1])] += freq
#         if not pairs:
#             break
#         most_frequent_pair = max(pairs, key=pairs.get)
#         merges.append(most_frequent_pair)
#         new_vocab = {}
#         for word, freq in vocab.items():
#             original_pair = " ".join(most_frequent_pair)
#             ###
#             # Insert your code here. Obtain the new word.
#             ###
#             new_pair = "".join(most_frequent_pair)
#             new_word = word.replace(original_pair, new_pair)  
#             #new_vocab[new_word] = freq
#         vocab = new_vocab
#     return merges

# corpus = ["low", "lowest", "lower", "low"]
# num_merges = 10
# merges = generate_merges(corpus, num_merges)

import spacy
import re
nlp = spacy.load('en_core_web_sm')
file = open("data/cs173_nlp_movie.txt", "rb")
text = file.read()
text = text.decode("utf-8")
print(len(re.findall(r"\bIron Man\b", text)))
