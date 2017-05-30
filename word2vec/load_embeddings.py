from gensim import models
import tensorflow as tf
import numpy as np

def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading external embeddings from %s" % path)
    # in original script provided by TAs, here was different method, but was depricated and killed script running
    model = models.Word2Vec.load(path)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0
    print("Vocab length is {}".format(len(vocab)))
    print("Model vocab length is {}".format(len(model.wv.vocab)))
    for idx, tok in enumerate(vocab.keys()):
        if tok in model.wv.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding}) # here, embeddings are actually set
