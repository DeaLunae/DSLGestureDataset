import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import DatasetCreator

DatasetCreator.create_validation_data_with_metadata()
df = DatasetCreator.load_data("GeneratedData\\ValidationWithMetaData")
df.drop(df[df['Success'] == "TRUE"].index, inplace=True)
df.drop(['Time', 'Success'], axis='columns', inplace=True)
index_columns = ['Gesture', 'Participant', 'Row', 'index']

df = df.rename(columns=lambda x: x.replace('Nummer', ''))
score_columns = [col for col in df.columns if col not in index_columns]

correlation_matrix = df[score_columns].corr()
plt.figure(figsize=(12, 9))  # You can change the size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',fmt=".2f", annot_kws={"size": 6},vmin=-0.1, vmax=0.1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.show()