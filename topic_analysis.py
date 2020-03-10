import glob
import re
import gensim
import numpy as np

files = glob.glob("./assets/*")
num_topics = 3


def analysis_topic():
    for file_paths in files:
        group_topic_model, dictionary = create_group_topic(file_paths)
        write_topic(file_paths, group_topic_model)
        fit_topic(file_paths, group_topic_model, dictionary)


def create_group_topic(file):
    file = open(file + "/stepGroupSizeSplit_Corpus.txt", "r")
    lines = file.readlines()
    data = []
    for line in lines:
        data.append(line.rstrip("\n").split(","))
    file.close()
    dictionary = gensim.corpora.Dictionary(data)
    # dictionary.filter_extremes(no_below=5, no_above=5)
    # create corpus
    corpus = [dictionary.doc2bow(words) for words in data]
    # LDAモデルの構築
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          num_topics=num_topics,
                                          id2word=dictionary,
                                          random_state=1)
    return lda, dictionary


def fit_topic(file_path, lda, dictionary):
    file = open(file_path + "/stepSplit_Corpus.txt", "r")
    lines = file.readlines()
    i = 1
    for line in lines:
        data = [line.rstrip("\n").split(",")]
        write_fit_topic(file_path, lda, dictionary, data, i)
        i += 1
    file.close()


def write_topic(file_path, lda):
    # ファイル書き込み
    for i in range(num_topics):
        file = open(file_path + "/group_topic" + str(i + 1) + ".txt", "w")
        for t in lda.show_topic(i):
            x = t[1]
            file.write(t[0] + "," + np.str(x))
            file.write("\n")
        file.close()


def write_fit_topic(file_path, lda, dictionary, data, i):
    file = open(file_path + "/fit_topic.csv", "w")
    corpus = [dictionary.doc2bow(words) for words in data]
    for topics_per_document in lda[corpus]:
        topic_text = re.sub(r"[ ()\[￿\]]", "", str(topics_per_document))
        file.write(str(i) + "," + topic_text)
        file.write("\n")
    file.close()


if __name__ == '__main__':
    analysis_topic()
