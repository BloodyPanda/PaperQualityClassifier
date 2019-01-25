# coding=utf-8
import sys
sys.path.append('..')
import mxnet as mx
import gluonnlp as nlp
import train_helper as th
import jieba
from mxnet import gluon, nd, init
import numpy as np
from process_data import get_data1, pad_sequences
from algorithms.bi_lstm import MyBiLSTM
from weighted_softmaxCE import WeightedSoftmaxCE
CTX = mx.gpu()
import random
random.seed(10)
np.random.seed(2018)
mx.random.seed(2018)
PAD = '<PAD>'
UNK = '<UNK>'
def get_vocab(sentences, customer_embedding_path, max_words):
    '''构建 Vocab，并对其附着词向量

    Args:
        sentences (list): 句子的列表
        customer_embedding_path (str): 预训练的词向量路径

    Returns:
        my_vocab (nlp.Vocab): 词典
    '''

    tokens = []
    for sent in sentences:
        tokens.extend(list(jieba.cut(sent)))  # 去掉\n,还原重新分词，将分好的词放入tokens中

    token_counter = nlp.data.count_tokens(tokens)  # 统计词频，输入词列表，输出词频键值对字典
    my_vocab = nlp.Vocab(token_counter)  # 加入unk="<unk>", reserved="['<pad>', '<bos>', '<eos>'形成词和整数的映射
    word_index = my_vocab.token_to_idx

    #将引入的词典装入字典数据结构中
    embeddings_index = {}
    f = open(customer_embedding_path)
    for line in f :
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    #建立预训练的词典

    embedding_dim = 300
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    #词向量路径
    #my_embedding = nlp.embedding.TokenEmbedding(unknown_token=UNK).from_file(customer_embedding_path)#要从自定义预训练令牌嵌入文件加载嵌入向量，请使用 gluonnlp.embedding.from_file()
    #my_vocab.set_embedding(my_embedding)#我们可以将单词嵌入customer_embedding_path中的sgns.baidubaike.bigram-char附加到索引单词 my_vocab

    return embedding_matrix#返回词向量被替换的词典

def sentences2idx(texts, my_vocab):
    '''句子的词转为索引

    Args:
        texts (list): 句子列表
        my_vocab (nlp.Vocab): 词典

    Returns:
        sentences_indices (list): 句子的词的索引的列表
    '''

    texts_indices = []
    for sent in texts:
        texts_indices.append(my_vocab.to_indices(list(jieba.cut(sent.strip().replace(' ', '')))))
    return texts_indices

def main():
    #content_path = '../data/paper_path_content10.txt'
    #category_path = '../data/paper_category10.txt'
    content_path = '../data/papertext.txt'
    category_path = '../data/papercategory.txt'

    titles, contents, labels = get_data1(content_path, category_path)
    from collections import Counter
    print(Counter(labels))


    #方法2 用别人的词典
    max_words = 10000
    customer_embedding_path = '../data/word_embedding/sgns.baidubaike.bigram-char'  # 引入预训练的词向量
    my_vocab = get_vocab(contents, customer_embedding_path, max_words)  # 调用get_vocab()返回字典
    pad_num_value = my_vocab.to_indices(PAD)#

    # 将输入数据转为整数索引
    input_idx = sentences2idx(contents, my_vocab)  # 调用sentences2idx()将句子列表转化成索引列表
    # 准备训练和验证数据迭代器
    max_seq_len = 10
    contents = pad_sequences(input_idx, max_seq_len, pad_num_value)  # 进行句子填充返回补短（PAD）截取的数据



    # 构建数据集
    dataset = gluon.data.SimpleDataset([[content, label] for content, label in zip(contents, labels)])
    train_dataset, valid_dataset = nlp.data.train_valid_split(dataset, 0.1)# 训练集：验证集 9:1
    train_dataset_lengths = [len(data[0]) for data in train_dataset]
    print(len(train_dataset), len(valid_dataset))
    print(len(train_dataset_lengths))

    # Bucketing 与 Dataloader
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(), nlp.data.batchify.Stack())
    batch_sampler = nlp.data.sampler.FixedBucketSampler(train_dataset_lengths, batch_size=32, num_buckets=10,
                                                        ratio=0.5, shuffle=True)
    train_dataloader = gluon.data.DataLoader(train_dataset, batch_sampler=batch_sampler, batchify_fn=batchify_fn)
    valid_dataloader = gluon.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, batchify_fn=batchify_fn)

    # 设置模型超参数并构建模型
    vocab_size = len(my_vocab) #词典长度
    word_vec_size = 300        #词向量维度
    nhidden_units = 128        #一层神经元个数
    nlayers = 2                #隐藏层层数
    drop_prob = 0.3            #梯度丢失
    nclass = 3                 #分3类，输出结点设置为3

    model = MyBiLSTM(vocab_size, word_vec_size, nhidden_units, nlayers, drop_prob, nclass)
    model.initialize(init=init.Xavier(), ctx=CTX)
    model.hybridize()
    # Attach a pre-trained glove word vector to the embedding layer
    model.embedding_layer.weight.set_data(my_vocab.embedding.idx_to_vec)
    # fixed the layer
    model.embedding_layer.collect_params().setattr('grad_req', 'null')

    # 定义损失函数与优化器
    nepochs, lr = 10, 0.001
    loss = WeightedSoftmaxCE()
    class_weight = nd.array([1, 1, 1], ctx=CTX)

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})

    # 训练
    th.train(train_dataloader, valid_dataloader, model, loss, class_weight, trainer, CTX, nepochs, clip=5.0)

    # 保存模型
    model_path = '../models/bi_lstm/bi_lstm_model'
    model.export(model_path)
    print('训练完成，模型已保存到: ', model_path)


if __name__ == '__main__':
    main()
