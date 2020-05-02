import os
import glob
import csv
import gensim
import numpy as np
import pyLDAvis.gensim

files_dir = glob.glob("./assets/*")


class AnalysisTopic:
    def __init__(self, files_path, num_topics=3, analysis_data_name="group_size_split_corpus.txt",
                 verification_data_name="step_split_corpus.txt"):
        self.files_path = files_path
        self.num_topics = num_topics
        self.analysis_data_name = analysis_data_name
        self.verification_data_name = verification_data_name
        self.dictionary = None
        self.corpus = None
        self.lda = None

    def create_topic(self):
        file = open(self.files_path + "/" + self.analysis_data_name, "r")
        lines = file.readlines()
        data = []
        for line in lines:
            data.append(line.rstrip("\n").split(","))
        file.close()
        self.dictionary = gensim.corpora.Dictionary(data)
        # dictionary.filter_extremes(no_below=5, no_above=5)
        # create corpus
        self.corpus = [self.dictionary.doc2bow(words) for words in data]
        # LDAモデルの構築
        self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                   num_topics=self.num_topics,
                                                   id2word=self.dictionary,
                                                   random_state=1)

    def _mkdir_out(self):
        out_file_path = self.files_path + "/topic_k" + str(self.num_topics)
        if not os.path.exists(out_file_path):
            os.mkdir(out_file_path)

    def write_topic(self):
        self._mkdir_out()
        out_file_path = self.files_path + "/topic_k" + str(self.num_topics)
        for i in range(self.num_topics):
            file = open(out_file_path + "/group_topic" + str(i + 1) + ".txt", "w")
            for t in self.lda.show_topic(i):
                x = t[1]
                file.write(t[0] + "," + np.str(x))
                file.write("\n")
            file.close()

    def write_fit_topic(self):
        self._mkdir_out()
        read_file = open(self.files_path + "/" + self.verification_data_name, "r")
        reader = csv.reader(read_file)
        out_file_path = self.files_path + "/topic_k" + str(self.num_topics)
        out_file = open(out_file_path + "/fit_topic.csv", "w")
        writer = csv.writer(out_file)
        csv_label = ["step"] + ["topic{}".format(tp + 1) for tp in range(self.num_topics)]
        writer.writerow(csv_label)
        for i, line in enumerate(reader):
            data = [line]
            corpus = [self.dictionary.doc2bow(words) for words in data]
            write_topic = [0] * (self.num_topics + 1)
            #  num step
            write_topic[0] = i + 1
            for topics_per_document in self.lda[corpus]:
                for topic in topics_per_document:
                    write_topic[topic[0] + 1] = topic[1]
            writer.writerow(write_topic)
        read_file.close()
        out_file.close()

    def write_LDAvis(self):
        self._mkdir_out()
        out_file_path = self.files_path + "/topic_k" + str(self.num_topics)
        vis_pcoa = pyLDAvis.gensim.prepare(self.lda, self.corpus, self.dictionary, sort_topics=False)
        pyLDAvis.save_html(vis_pcoa, out_file_path + '/pyldavis_output_pcoa.html')


if __name__ == '__main__':
    for a_files_path in files_dir:
        analysis_topic = AnalysisTopic(a_files_path, 3)
        analysis_topic.create_topic()
        analysis_topic.write_topic()
        analysis_topic.write_fit_topic()
        analysis_topic.write_LDAvis()
