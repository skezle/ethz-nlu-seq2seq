import gensim
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TRAIN_FILE = "data/Training_Shuffled_Dataset.txt"
VALIDATION_FILE = "data/Validation_Shuffled_Dataset.txt"
WORD2VEC = "word2vec/wordembeddings_200.word2vec"

EMBEDDING_SIZE = 200
MINIMAL_WORD_FREQUENCY = 1


def load_sentences(train_path, validation_path=None):
    sentences = []

    f = open(train_path, 'r')
    for line in f:
        conversation = line.strip().split('\t')
        for i in range(3):
            sentence = []
            for word in conversation[i].split():
                sentence.append(word)
            sentences.append(sentence)

    if validation_path is not None:
        f = open(VALIDATION_FILE, 'r')
        for line in f:
            conversation = line.strip().split('\t')
            for i in range(3):
                sentence = []
                for word in conversation[i].split():
                    sentence.append(word)
                sentences.append(sentence)

    return sentences


def train_embeddings(save_to_path, embedding_size, minimal_frequency, train_path, validation_path=None, num_workers=4):
    print("Training word2vec model with following parameters: ")
    print("\t dataset: " + train_path + ", validation set used: " + validation_path)
    print("\t word embeddings size: " + str(embedding_size))
    print("\t word embeddings saved on path: {}".format(save_to_path))
    model = gensim.models.Word2Vec(load_sentences(train_path, validation_path),
                                   size=embedding_size,
                                   min_count=minimal_frequency,
                                   workers=num_workers)
    model.save(save_to_path)


def evaluate(path):

    print("Loading word2vec model from {}".format(path))
    model = gensim.models.Word2Vec.load(path)
    print('Vocabulary size is: {}'.format(len(model.wv.vocab)))
    print(model.most_similar(
        positive=['woman', 'king'], negative=['man'], topn=5))
    print(model.most_similar(
        positive=['girl', 'man'], negative=['boy'], topn=5))
    print(model.most_similar(
        positive=['head', 'eye', 'lips', 'nose'], negative=['toes'], topn=5))
    print(model.most_similar(
        positive=['pain'], topn=10))
    print(model.most_similar(
        positive=['pain'], negative=['happy'], topn=10))

    x = model[model.wv.vocab]
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x[:5000,:])

    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(x[:5000, :])

    plt.scatter(x_pca[:, 0], x_pca[:, 1])
    for label, x, y in zip(model.wv.vocab, x_pca[:, 0], x_pca[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    for label, x, y in zip(model.wv.vocab, x_tsne[:, 0], x_tsne[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def main():
    # train_embeddings(WORD2VEC)
    evaluate(WORD2VEC)

if __name__ == "__main__":
    main()



