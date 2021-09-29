import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt


def load_from_file(_path: str) -> dict:
    _inp = pd.read_csv(_path, encoding="cp1251")

    _ham = []
    for line in _inp[_inp.v1 == "ham"].values:
        _tmp = ""
        for item in line[1:]:
            if isinstance(item, str):
                _tmp += item
        _ham.append(_tmp)

    _spam = []
    for line in _inp[_inp.v1 == "spam"].values:
        _tmp = ""
        for item in line[1:]:
            if isinstance(item, str):
                _tmp += item
        _spam.append(_tmp)

    return {"ham": _ham, "spam": _spam}


def clean_str(_inp: str) -> str:
    _out = ""
    for char in _inp:
        if char.isalpha():
            _out += char.lower()
        else:
            _out += " "
    return _out


def tokenization(_inp: str) -> list:
    ls = []
    for word in _inp.split(" "):
        if word != "":
           ls.append(word)
    return ls


def remove_stopwords(_tokens: list) -> list:
    stops = stopwords.words("english")
    _out = [token.strip() for token in _tokens if token not in stops]
    return _out


def stemming(_inp: list) -> list:
    ps = PorterStemmer()
    arr = []
    for word in _inp:
        arr.append(ps.stem(word))
    return arr


def build_plot(_ham, _spam):
    _, axs = plt.subplots(2, 2)

    word_len_ham = count_len_words(to_single_list(_ham))
    word_len_spam = count_len_words(to_single_list(_spam))

    message_len_ham = count_message_len(to_single_list(_ham))
    message_len_spam = count_message_len(to_single_list(_spam))

    num_words_ham = find_top_20(count_words(to_single_list(_ham)))
    num_words_spam = find_top_20(count_words(to_single_list(_spam)))

    dict_list = list(num_words_ham.items())
    dict_list.sort(key=lambda i: i[1], reverse=True)
    new_num_words_ham = dict(dict_list)

    axs[0, 0].bar(word_len_ham.keys(), normalize(word_len_ham), color="y", alpha=0.5)
    axs[0, 0].bar(word_len_spam.keys(), normalize(word_len_spam), color="b", alpha=0.5)
    axs[0, 0].legend(["ham", "spam"])
    axs[0, 0].set_xlabel("size")
    axs[0, 0].set_ylabel("words")

    axs[0, 1].bar(message_len_ham.keys(), message_len_ham.values(), color="y", alpha=0.5)
    axs[0, 1].bar(message_len_spam.keys(), message_len_spam.values(), color="b", alpha=0.5)
    axs[0, 1].legend(["ham", "spam"])
    axs[0, 1].set_xlabel("size")
    axs[0, 1].set_ylabel("messages")

    axs[1, 0].bar(new_num_words_ham.keys(), new_num_words_ham.values(), color="y", alpha=0.5)
    axs[1, 0].legend(["ham"])
    axs[1, 0].set_xlabel("word")
    axs[1, 0].set_ylabel("numbers")

    axs[1, 1].bar(num_words_spam.keys(), num_words_spam.values(), color="b", alpha=0.5)
    axs[1, 1].legend(["ham", "spam"])
    axs[1, 1].set_xlabel("size")
    axs[1, 1].set_ylabel("words")

    plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()


def get_av(_inp: dict) -> int:
    all = 0
    for k, v in _inp.items():
        all += k*v
    return all/sum(_inp.values())


def to_single_list(_inp: list) -> list:
    _arr = []
    for subarr in _inp:
        _arr.extend(subarr)

    return _arr


def count_words(_inp: list) -> dict:
    """Note: need to_single_list before"""
    _dict = dict()
    for word in _inp:
        try:
            _dict[word] += 1
        except KeyError:
            _dict[word] = 1

    return _dict


def count_len_words(_inp: list) -> dict:
    """Note: need to_single_list before"""
    _dict = dict()
    for word in _inp:
        try:
            _dict[len(word)] += 1
        except KeyError:
            _dict[len(word)] = 1

    return _dict


def count_message_len(_inp: list) -> dict:
    _dict = dict()
    for message in _inp:
        try:
            _dict[len(message)] += 1
        except KeyError:
            _dict[len(message)] = 1

    return _dict


def normalize(_inp: dict) -> list:
    s = 0
    ls = []
    for k, v in _inp.items():
        s += k*v

    for i in _inp.values():
        ls.append(i/s)

    return ls


def find_top_20(_inp: dict) -> dict:
    new_dict = dict()
    tmp = list(_inp.values())
    tmp.sort()
    tmp.reverse()
    tmp = tmp[:20]
    for k in _inp.keys():
        if _inp[k] in tmp:
            new_dict[k] = _inp[k]

    return new_dict


if __name__ == '__main__':
    ham = []
    spam = []

    categories = load_from_file("./input/sms-spam-corpus.csv")
    for line in categories["ham"]:
        ham.append(stemming(remove_stopwords(tokenization(clean_str(line)))))

    for line in categories["spam"]:
        spam.append(stemming(remove_stopwords(tokenization(clean_str(line)))))

    df = pd.DataFrame.from_dict(count_words(to_single_list(ham)), orient="index")
    df.to_csv("output/ham.csv")
    df = pd.DataFrame.from_dict(count_words(to_single_list(spam)), orient="index")
    df.to_csv("output/spam.csv")
    build_plot(ham, spam)
