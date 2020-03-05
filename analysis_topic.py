import gensim
import numpy as np

def create_corpus():
    test_data = open("assets/corpus.txt", "r")
    lines = test_data.readlines()
    data = []
    for line in lines:
        data.append(line.rstrip("\n").split(","))
    test_data.close()
    dictionary = gensim.corpora.Dictionary(data)
    #dictionary.filter_extremes(no_below=5, no_above=5)

    # create corpus
    corpus = [dictionary.doc2bow(words) for words in data]

    # LDAモデルの構築
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          num_topics=3,
                                          id2word=dictionary,
                                          random_state=1)
    select(lda, dictionary)
    write(lda)

def select(lda, dictionary):
    test_data = open("assets/corpus_step.txt", "r")
    lines = test_data.readlines()
    i = 1
    for line in lines:
        data = []
        data.append(line.rstrip("\n").split(","))
        show(lda, dictionary, data, i)
        i += 1
    test_data.close()

def show(lda, dictionary, data, i):
    corpus = [dictionary.doc2bow(words) for words in data]
    for topics_per_document in lda[corpus]:
        print(str(i) + "秒 = " +str(topics_per_document))

def write(lda):
    # ファイル書き込み
    file = open("out/topic.txt", "w")
    for t in lda.show_topic(1):
        x = t[1]
        file.write(t[0] + "," + np.str(x))
        file.write("\n")
    file.close()

if __name__ == '__main__':
    create_corpus()
