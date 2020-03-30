import glob
import csv
import gensim
import numpy as np

files = glob.glob("./assets/*")
num_topics = 3


def analysis_topic():
    for file_paths in files:
        group_topic_model, dictionary = create_group_topic(file_paths)
        write_topic(file_paths, group_topic_model)
        write_fit_topic(file_paths, group_topic_model, dictionary)


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


def write_topic(file_path, lda):
    # ファイル書き込み
    for i in range(num_topics):
        file = open(file_path + "/group_topic" + str(i) + ".txt", "w")
        for t in lda.show_topic(i):
            x = t[1]
            file.write(t[0] + "," + np.str(x))
            file.write("\n")
        file.close()


def write_fit_topic(file_path, lda, dictionary):
    read_file = open(file_path + "/stepSplit_Corpus.txt", "r")
    reader = csv.reader(read_file)
    out_file = open(file_path + "/fit_topic.csv", "w")
    writer = csv.writer(out_file)
    csv_label = ["step"]
    csv_label.extend(["topic{}".format(tp) for tp in range(num_topics)])
    writer.writerow(csv_label)
    for i, line in enumerate(reader):
        data = [line]
        corpus = [dictionary.doc2bow(words) for words in data]
        write_topic = [0] * (num_topics + 1)
        write_topic[0] = i + 1
        for topics_per_document in lda[corpus]:
            for topic in topics_per_document:
                write_topic[topic[0]+1] = topic[1]
        writer.writerow(write_topic)
    read_file.close()
    out_file.close()


if __name__ == '__main__':
    analysis_topic()
