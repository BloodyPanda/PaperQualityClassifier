# -*- coding=utf-8 -*-
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.layers.core import Activation, Dense, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from model import *
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

# some config values
MAX_FEATURES = 40000                         # 要用多少个独立的词（num rows in embedding vector）
MAX_SENTENCE_LENGTH = 2000                   # 一个句子最多使用多少个词
NB_CLASSES = 3                               # 类别个数
EMBEDDING_SIZE = 300                         # 词向量大小，之后在读入词典时会改变
content_path = '../data/papertext.txt'       # 文件内容所在路径
category_path = '../data/papercategory.txt'  # 文件标题标签所在路径
EMBEDDING_FILE = '../data/word_embedding/sgns.baidubaike.bigram-char'  # 词典路径


def get_data(content_path, category_path):
    titles, contents, labels = [], [], []
    with open(content_path, 'r', encoding='utf-8') as fr:
        for paper in fr:
            title, content = paper.strip().split('|||')
            titles.append(title)

            # 取消别人的切词
            # contents.append(content.replace(' ', ''))
            contents.append(content)

    with open(category_path, 'r', encoding='utf-8') as frc:
        for info in frc:
            _, _, label = info.strip().split('|||')
            labels.append(int(label))
    a = len(titles)
    b = len(contents)
    c = len(labels)

    assert a == b == c
    # assert len(titles) == len(contents) == len(labels)

    return titles, contents, labels


def load_and_prec():
    titles, contents, labels = get_data(content_path, category_path)
    from collections import Counter
    print(Counter(labels))

    # 因为语料是分好词的文本，直接调用tokenizer进行分词
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(contents)
    train_X = tokenizer.texts_to_sequences(contents)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=MAX_SENTENCE_LENGTH)
    train_y = labels

    # shuffling the data
    np.random.seed(2019)
    trn_idx = np.random.permutation(len(train_X))

    train_X = np.array(train_X)[trn_idx]
    train_y = np.array(train_y)[trn_idx]
    train_X = np.expand_dims(train_X, axis=1)

    train_y = to_categorical(train_y, NB_CLASSES)  # 转换成独热编码的形式

    return train_X, train_y, tokenizer.word_index


def load_vocab(word_index):
    # 将引入的词典装入字典数据结构中
    embeddings_index = {}
    f = open(EMBEDDING_FILE, encoding='utf-8')
    l_c = 0
    lines_num = 0
    global EMBEDDING_SIZE

    for line in f:
        values = line.strip().split(' ')
        if l_c == 0:
            lines_num = int(values[0])
            EMBEDDING_SIZE = int(values[1])
            l_c += 1
            continue
        word = values[0]
        if len(values[1:]) != EMBEDDING_SIZE:
            continue
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def main():
    train_X, train_y, word_index = load_and_prec()

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=0)

    embedding_matrix = load_vocab(word_index)  # 调用get_vocab()返回字典

    HIDDEN_LAYER_SIZE = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    model = createHierarchicalAttentionModel(MAX_SENTENCE_LENGTH, embWeights=embedding_matrix,
                                             embeddingSize=EMBEDDING_SIZE,
                                             vocabSize=min(MAX_FEATURES, len(word_index) + 1),
                                             class_num=NB_CLASSES, last_activation='softmax',
                                             loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_X, train_y, batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_split=0.2)

    # plot loss and accuracy
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # 评分
    score, acc = model.evaluate(test_X, test_y, batch_size=BATCH_SIZE)
    print("Test score: %.3f, accuracy: %.3f" % (score, acc))

if __name__ == '__main__':
    main()