import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    cümleyi kelime / simge dizisine bölme bir simge, 
    bir kelime veya noktalama karakteri veya sayı olabilir
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    kök bulma = kelimenin kök biçimini bul
    ornek:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    bag of words dizisinin dönüş:
    Cümledeki bilinen her kelime için 1, aksi takdirde 0
    ornek:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # her kelimenin kökü
    sentence_words = [stem(word) for word in tokenized_sentence]
    # torbayı her kelime için 0 ile başlat
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag