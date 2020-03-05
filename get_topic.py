import gensim
import numpy as np

test_data = open("assets/corpus.txt", "r")
lines = test_data.readlines()

data = []
for line in lines:
    data.append(line.rstrip("\n").split(","))
test_data.close()

dictionary = gensim.corpora.Dictionary(data)
#dictionary.filter_extremes(no_below=10, no_above=10)


# create corpus
corpus = [dictionary.doc2bow(words) for words in data]

# learn topic
lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                      num_topics=3,
                                      id2word=dictionary,
                                      random_state=1)

#print('topics: {}'.format(lda.show_topics()))
#print(lda.show_topics())

# # 見やすく出力
# for i in range(2):
#     print("\n")
#     print("="*80)
#     print("TOPIC {0}\n".format(i))
#     topic = lda.show_topic(i)
#     for t in topic:
#         #print("{0:20s}{1}".format(t[0], t[1]))
#         print(t[0])


#ファイル書き込み
# file = open("out/topic.txt", "w")
file = open("/Users/ryusei/IdeaProjects/Pedestrian_Simulation/core/assets/topic.txt", "w")
for t in lda.show_topic(2):
    x = t[1]
    file.write(t[0] + "," + np.str(x))
    file.write("\n")
file.close()