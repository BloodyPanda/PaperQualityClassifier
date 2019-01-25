import numpy as np


def get_data1(content_path, category_path):
    titles, contents, labels = [], [], []
    with open(content_path, 'r', encoding='utf-8') as fr:
        for paper in fr:
            title, content = paper.strip().split('|||')
            titles.append(title)
            content = content[::-1]
            # 取消别人的切词
            contents.append(content.replace(' ', ''))

    with open(category_path, 'r', encoding='utf-8') as frc:
        for info in frc:
            _, _, label = info.strip().split('|||')
            labels.append(int(label))
    a = len(titles)
    b = len(contents)
    c = len(labels)

    assert a == b == c
    return titles, contents, labels


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
    #assert len(titles) == len(contents) == len(labels)

    return titles, contents, labels

def pad_sequences(sequences, max_len, pad_value):
    """对句子进行长截短补到指定的长度

    Args:
        sequences (列表的列表): ，内层列表存放每个句子的切词的结果
        max_len (int): 最大长度
        pad_value (int): 要填补的值

    Returns:
        paded_seqs (np.array()): 填补或截断后的数据
    """

    # max_len = max(map(lambda x: len(x), sequences))

    paded_seqs = np.zeros((len(sequences), max_len))
    for idx, seq in enumerate(sequences):
        paded = None
        if len(seq) < max_len:
            paded = np.array(([pad_value] * (max_len - len(seq)) + seq))
        else:
            paded = np.array(seq[0:max_len])
        paded_seqs[idx] = paded

    return paded_seqs


if __name__ == "__main__":
    content_path = '../data/paper_path.txt'
    category_path = '../data/paper_category.txt'

    titles, contents, labels = get_data(content_path, category_path)
    print(labels)
