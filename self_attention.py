import torch
import torch.nn.functional as F

# Let's map each unique word to the index in which that word occurs
sentence = 'Life is short, eat dessert first'
sentence = sentence.replace(',', '').split()
dict_words = {word: idx for idx, word in enumerate(sorted(sentence))}
sentence_ints = torch.tensor([dict_words[char] for char in sentence])

# Now create a n_words x vec_dim = 6 x 16 embedding tensor (random initialized values)
torch.manual_seed(42)
dim_word_embedding = 16
embedding_layer = torch.nn.Embedding(6, dim_word_embedding)
embedded_sentence = embedding_layer(sentence_ints).detach()
print('Embedded sentence shape:', embedded_sentence.shape)

# The projection (i.e. weights) matrices for queries, keys and values are updated during the training process
# x_i is a token in the input sentence, whose length is T. Thus, 1 <= i <= T.
# 
# q_i = W_q @ x_i 
# k_i = W_k @ x_i
# v_i = W_v @ x_i
# 
# ...
# 
# q_T = W_q @ x_T 
# k_T = W_k @ x_T
# v_T = W_v @ x_T


dim_q, dim_k, dim_v = 24, 24, 28 # dim_q and dim_k MUST be equal
W_q, W_k, W_v = torch.rand(dim_q, dim_word_embedding), torch.rand(dim_k, dim_word_embedding), torch.rand(dim_v, dim_word_embedding)

# Let's compute query, key, and value for the first word in the sentence
word_idx = 0
x_1 = embedded_sentence[word_idx]
query_1 = W_q @ x_1
key_1 = W_k @ x_1
value_1 = W_v @ x_1
print(f'Word idx = {word_idx}\n', 'Query size:', query_1.shape, '\n', 
      'Key size:', key_1.shape, '\n', 'Value size:', value_1.shape, '\n')

# Now that we have understood the process, let's extend this to all the inputs
# Transpose the matrix to have words on rows and vector components on columns
keys = (W_k @ embedded_sentence.T).T
values = (W_v @ embedded_sentence.T).T
print("keys.shape:", keys.shape) 
print("values.shape:", values.shape)

# Let's compute the unnormalized attention score for the first word w.r.t. the 5th word
omega_1_4 = query_1 @ keys[4]
print(omega_1_4)

# Let's compute the unnormalized attention score for the first word w.r.t. to all the other words
omega_1_all = query_1 @ keys.T
print(omega_1_all)

# Why unnormalized attention score? Because they must be normalzied in a way. Which one? Softmax + scaling (for avoiding numeric instability)
attention_weights_1 = F.softmax(omega_1_all / dim_k ** 0.5, dim=0)
print(attention_weights_1)

# Compute z, an enhanced (contains the information about ALL the other words) version of the input word
# The dimension is dim_v (in this case higher than the original input, but can be chosen freely)
context_vector_z_1 = attention_weights_1 @ values
print(context_vector_z_1.shape)
print(context_vector_z_1)

# Multi-head
n_heads = 3
multihead_W_query = torch.rand(n_heads, dim_q, dim_word_embedding) 
multihead_W_key = torch.rand(n_heads, dim_k, dim_word_embedding)
multihead_W_value = torch.rand(n_heads, dim_v, dim_word_embedding)

# Each query now has size 3 x dim_word_embeddings
multihead_query_1 = multihead_W_query @ x_1
print(multihead_query_1.shape)

# Stack input to make it feasible as input
stacked_inputs = embedded_sentence.T.repeat(3, 1, 1)
print(stacked_inputs.shape)

# Batch matrix multiplication 
multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)
multihead_values = torch.bmm(multihead_W_value, stacked_inputs)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)

# Permute the last two dimensions
# After this, the first represents attention head, the second one the word, the third the vector components
multihead_keys = multihead_keys.permute(0, 2, 1)
multihead_values = multihead_values.permute(0, 2, 1)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)