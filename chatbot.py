from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import csv
from nltk.corpus import stopwords
import random
def Chatbot(sent):
    synonyms = {"temperature": ["weather"],
                "who":["what","how","when","which batsman"],
                "how": ["what", "whwn", "when"],
                "when": ["what", "how", "who"],
                "what": ["who", "how", "when"],
                "which batsman":["who"]}
    stopWords = set(stopwords.words('english'))
    liss=sent.split()
    fin_array=[]
    for iter in liss:
        if iter not in stopWords:
            fin_array.append(iter)
    sent= " ".join(fin_array)
    csv_file_path = "Chatbot.csv"
    i = 0
    words = []
    words.append(" BJB ")
    words.append(" BJB4 ")
    test_set = [sent]
    sent=sent.split()
    for each in range(len(sent)):
        if sent[each] in synonyms:
            for items in synonyms[sent[each]]:
                test_set.append(" ".join(sent[0:each] + [items] + sent[each + 1:len(sent)]))

    with open(csv_file_path, "r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        for row in reader:
            exempted=[]
            data=word_tokenize(row[0])
            for w in data:
                if w not in stopWords:
                    exempted.append(w)

            words.append(" ".join(exempted))
            i += 1
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(words)  # finds the tfidf score with normalization

    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)

    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

    cosine = np.delete(cosine, 0)
    max = cosine.max()
    response_index = 0
    if (max > 0.5):
        new_max = max - 0.01
        list = np.where(cosine == max)
        response_index = random.choice(list[0])

    else:
        return None

    j = 0
    with open(csv_file_path, "r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        for row in reader:
            j += 1  # we begin with 1 not 0 &    j is initialized by 0
            if j == response_index:

                return row[1], response_index,
                break
while 1:
    sentence = input("Please Enter: ")
    if sentence.lower()=="bye":
        break
    reply=Chatbot(sentence)
    if reply:
        response_primary, line_id_primary = Chatbot(sentence)
        print(response_primary)
    else:
        print("Sorry I dont Understand. Can you please repeat")
