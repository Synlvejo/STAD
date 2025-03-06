import json
import pandas as pd
import itertools
from gensim.models import Word2Vec
import argparse

parser = argparse.ArgumentParser(description="preprocess...")
parser.add_argument('-p', '--program', type=str, required=True, help='program')
parser.add_argument('-d', '--duration', type=str, required=True, help='duration')
parser.add_argument('-n', '--num', type=int, required=True, help='nodes num')

args = parser.parse_args()

program = args.program
duration = args.duration
num = args.num
dimension = 32

vec_input_path = "/path/to/MPI_profile/"+ program + "/" + duration + "/graph"
vec_output_path = "/path/to/MPI_profile/"+ program + "/" + duration + "/node_feature.csv"

print("vectorizeing ...")

data = pd.read_csv(vec_input_path, header=0)
print(len(data))

sentences = data.reset_index(drop=False).apply(lambda row: row.tolist(), axis=1).tolist()
model = Word2Vec(sentences, vector_size=dimension, window=5, min_count=1, workers=4)

def get_vector(row):
    vectors = [model.wv[str(num)] for num in row if str(num) in model.wv]
    if vectors:
        return sum(vectors) / len(vectors)  

vectors = [get_vector(row) for row in sentences]

df_vectors = pd.DataFrame(vectors, columns=[f'vector_{i}' for i in range(dimension)])
df_vectors = pd.concat([data.iloc[:, 0], data.iloc[:, 1], df_vectors], axis=1)

def filter_tid(group, tid):
    filtered = group[group['tid'] == tid]
    if filtered.empty:
        filtered = pd.DataFrame({'ts_id': group.name, 'tid': [tid]})
        for i in range(32):
           filtered[f'vector_{i}'] = [0]
    return filtered

results = []
for tid in range(num):
    df_vectors_i = df_vectors.groupby('ts_id').apply(filter_tid, tid)
    df_vectors_i = df_vectors_i.reset_index(drop=True)
    results.append(df_vectors_i.iloc[:, 2:])
    df_combined = pd.concat(results, axis=1)
    df_combined = df_combined.reset_index(drop=True)

nodes = [f'node_{i}' for i in range(num)]
vectors = [f'vector_{i}' for i in range(dimension)]

repeated_nodes = list(itertools.repeat(node, dimension) for node in nodes)
repeated_vectors = list(itertools.repeat(vectors, num))

nodes = [item for sublist in repeated_nodes for item in sublist]
vectors = [item for sublist in repeated_vectors for item in sublist]

df_metric = pd.DataFrame(data=[nodes], columns=vectors)

df_result = pd.concat([df_metric, df_combined], ignore_index=True)

df_result.to_csv(vec_output_path, header=True, index=False)


