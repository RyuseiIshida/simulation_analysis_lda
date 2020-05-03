import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("fit_topic.csv", sep=",")

sns.set()
sns.set_style('white')

fig = plt.figure()
bar_width = 0.8 / df.shape[1]
for i, y in enumerate(range(df.shape[1] - 1)):
    label = 'topic' + str(i + 1)
    plt.bar(df['step'] + i * bar_width, df[label], width=bar_width, align="center", label=label)
plt.xlabel("step")
plt.xticks(df['step'].values)
plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
plt.subplots_adjust(right=0.8)
fig.savefig('fit_topic.png')