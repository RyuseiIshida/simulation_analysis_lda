import pandas as pd
import seaborn as sns

# CSV読み込み
df = pd.read_csv("fit_topic.csv", sep=",")
df.columns = ["step", "topic1", "topic2", "topic2"]