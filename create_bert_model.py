from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('bert-base-nli-mean-tokens')
with open('bert_embedder.pkl', 'wb') as f:
    pickle.dump(model, f)