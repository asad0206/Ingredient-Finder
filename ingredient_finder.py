import numpy as np
import pandas as pd
import itertools
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

df = pd.concat([pd.read_json('C:\\Users\\Asad Ansari\\Desktop\\cooking project\\train.json'), pd.read_json('C:\\Users\\Asad Ansari\\Desktop\\cooking project\\test.json')]).reset_index()

df.ingredients = df.ingredients.apply(lambda ings : [ing.lower() for ing in ings])


cooc_counts = Counter()
ing_count  = Counter()
for ingredients in df.ingredients:
    for ing in ingredients:
        ing_count[ing] += 1
    for (ing_a, ing_b) in itertools.combinations(set(ingredients), 2):
        # NOTE: just making sure we added pairs in a consistent order (a < b); you can also add both (a,b) and (b,a) if you want.
        if ing_a > ing_b:
            ing_a, ing_b = ing_b, ing_a
        cooc_counts[(ing_a, ing_b)] += 1

cooc_df = pd.DataFrame(((ing_a, ing_b, ing_count[ing_a], ing_count[ing_b], cooc) for (ing_a, ing_b), cooc in cooc_counts.items()), columns=['a', 'b', 'a_count', 'b_count', 'cooc'])
cooc_df.sample(10)

cooc_df[cooc_df.a == 'chillies'].sort_values('cooc', ascending=False).head(10)

p_a = cooc_df.a_count / sum(ing_count.values())
p_b = cooc_df.b_count / len(ing_count.values())
p_a_b = cooc_df.cooc / cooc_df.cooc.sum()
cooc_df['pmi'] = np.log(p_a_b / (p_a * p_b))

cooc_df.sort_values('pmi', ascending=False).head(10)

min_count = 5
cooc_df[(cooc_df.a_count >= min_count) & (cooc_df.b_count >= min_count)].sort_values('pmi', ascending=False).head(20)

min_count = 30
cooc_df[(cooc_df.a_count >= min_count) & (cooc_df.b_count >= min_count)].sort_values('pmi', ascending=False).head(20)

data_df = cooc_df[cooc_df.pmi > 0].copy()
# Since the matrix is symetric, we add the same values for (b,a) as we have for (a,b)
data_df_t = data_df.copy()
data_df.a, data_df.b = data_df.b, data_df.a
data_df = pd.concat([data_df, data_df_t])

rows_idx, row_keys = pd.factorize(data_df.a)
cols_idx, col_keys = pd.factorize(data_df.b)
values = data_df.pmi

matrix = csr_matrix((values, (rows_idx, cols_idx)))
key_to_row = {key: idx for idx, key in enumerate(row_keys)}

svd = TruncatedSVD(200)
factors = svd.fit_transform(matrix)

def most_similar(ingredient, topn=10):
    if ingredient not in key_to_row:
        print("Unknown ingredient.")
    factor = factors[key_to_row[ingredient]]
    cosines = cosine_similarity([factor], factors)[0]
    indices = cosines.argsort()[::-1][:topn + 1]
    keys = [row_keys[idx] for idx in indices if idx != key_to_row[ingredient]]
    return keys, cosines[indices]

def display_most_similar(ingredient, topn=10):
    print("- Most similar to '{}'".format(ingredient))
    for similar_ing, score in zip(*most_similar(ingredient, topn)):
        print("  . {} : {:.2f}".format(similar_ing, score))

#display_most_similar('chile powder')

display_most_similar('noodles')
