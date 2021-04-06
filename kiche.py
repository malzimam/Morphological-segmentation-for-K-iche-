"""
Usage:
  kiche.py (--train=<train_data>) (--predict=<predict_data>) (--output=<out>)
  kiche.py (-h | --help)
  kiche.py --version

Options:
  -h --help     Show this screen.
  --train=<train_data>  train on a .tsv file
  --predict=<predict_data>  predict .tsv file
  --output=<out>  write to a .tsv file

"""

from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import re
from sklearn.svm import LinearSVC

def tokenizer(inp: object):
    """tokenize a pandas column of strings and add begnning and end markers
    :param inp: pandas column
    
    >>>tokenizer(df["foo"])
    [["^foo$", "^bar$"], ["^baz$"]]
    """
    words = [("^" + i + "$") for i in inp]
    for i in range(len(words)):
        words[i] = re.sub(" ", "$ ^", words[i])
        words[i] = words[i].split(' ')
    return words

def preprocess(inp: list):
    """turns list of lists of lexeme strings into a list of lists of lists of phoneme feature dictionaries
    :param inp: list to process
    
    >>> preprocess([["^abc$"]])
    """
    if not ( isinstance(inp, list) 
            and isinstance(inp[0], list)
            and isinstance(inp[0][0], str)):
        raise TypeError("input needs to be a list of lists of strings")
        
    stats = []
    vowels = {"u", "i", "a", "o", "e", "U", "I", "A", "O", "E", "ó"}
    velar = {"k", "x", "q", "K", "X", "Q", "h", "H"}
    for sent in inp:
        snt = []
        for index in range(len(sent)):
            word = sent[index]
            true_len = len(word) - 2
            if true_len <= 1:
                continue
            wrd = []
            for i in range(len(word))[1:true_len]:
                dic = {}
                dic["beg"]=i
                dic["sym"]=word[i]
                if dic["sym"] in vowels:
                    dic["vow"] = 1
                else:
                    dic["vow"] = 0
                if dic["sym"] in velar:
                    dic["vel"] = 1
                else:
                    dic["vel"] = 0
                dic["logpos"] = np.log1p(dic["beg"])
                dic["end"]=true_len - dic["beg"]
                dic["len"]=true_len
                dic["mult"]=np.log1p(dic["beg"] * dic["end"])
                dic["add"]=dic["len"] / dic["mult"]
                dic["relBeg"]=dic["beg"] / dic["len"]
                dic["relEnd"]=dic["end"] / dic["len"]
                dic["logBeg"]=np.log1p(dic["relBeg"])
                dic["logEnd"]=np.log1p(dic["relEnd"])
                dic["prev"]=word[i-1]
                if dic["prev"] in vowels:
                    dic["prevVowel"] = 1
                else:
                    dic["prevVowel"] = 0
                if dic["prev"] in velar:
                    dic["prevVelar"] = 1
                else:
                    dic["prevVelar"] = 0
                dic["next"]=word[i+1]
                if dic["next"] in vowels:
                    dic["nextVowel"] = 1
                else:
                    dic["nextVowel"] = 0
                if dic["next"] in velar:
                    dic["nextVelar"] = 1
                else:
                    dic["nextVelar"] = 0
                dic["next2"]=word[i+2]
                dic["nextBi"]=dic["next"] + dic["next2"]
                dic["prevBi"]=dic["prev"] + dic["sym"]
                dic["curBi"]=dic["sym"] + dic["next"]
                dic["prevTri"]=dic["prev"] + dic["curBi"]
                dic["curTri"]=dic["sym"] + dic["nextBi"]
                if "$" in dic["nextBi"]:
                    dic["nextTri"]=dic["nextBi"]+"$"
                else:
                    dic["nextTri"]=dic["nextBi"]+word[i+3]
                dic["curQuat"]=dic["prev"] + dic["curTri"]
                wrd.append(dic)
            snt.append(wrd)
        stats.append(snt)
    return stats

def tyndices(inp: str):
    """
    Returns the indices of morpheme boundaries for a given string as a set
    :param inp: the string to extract from
    """
    if not isinstance(inp, str):
        raise TypeError("Can only process strings")
    tyndex = 0
    ret = set()
    for i in range(len(inp)):
        if inp[i] == ">":
            ret.add(tyndex - 1)
        else:
            tyndex += 1
    return ret

def appendix(inp: list, tynd: set):
    """
    Adds a boundary parameter to the "bound" column of the specified dataframe
    :param inp: the pandas dataframe to add to
    :param tynd: a binary mapping of the boundary parameter for all entries in the dataframe
    """
    for index in range(len(inp)):
        if index in tynd:
            inp[index]["bound"] = 1
        else:
            inp[index]["bound"] = 0

def sent_process(inp: list, ind: int, target:object):
    tynd = 0
    for word_ind in range(len(inp)):
        wrd = inp[word_ind][:-1]
#         wrd = inp[word_ind]
        if len(wrd) == 0:
            continue
        hlp = tyndices(wrd)
#         print(tynd)
#         print(len(sym_stats[ind]))
#         print(sym_stats[ind][tynd])
        appendix(target[ind][tynd], hlp)
        tynd += 1            

def to_sound_list(inp: list): 
    all_sents = []
    for i in inp:
        all_sents += i
    all_words = []
    for i in all_sents:
        all_words += i
    return all_words        

def process_string(inp: str, encoder: object, model: object):
    """return the predicted version of the string
    :param inp: string to process
    :param encoder: one-hot encoder conditioned on the training set
    :param model: a linear regression model conditioned on the training set
    
    >>> process_string("acab")
    "ac>ab"
    """
    if not type(inp) == str:
        raise TypeError("only accepts strings as input")
#     go go ternary operators
    lookforinit = re.search(r"^\W+", inp)
    init = lookforinit[0] if lookforinit else None
    inp = inp[len(init):] if init else inp
    lookforend = re.search(r"\W+$", inp)
    after = lookforend[0] if lookforend else None
    inp = inp[:-len(after)] if after else inp
    
    delimiters = re.findall(r"( [a-zA-Z][.,?:;¿¡\-% ]+|[^a-zA-Zóʼ]+)", inp)
    inp_new = "^" + re.sub(" ", "$ ^", inp) + "$"
    inp_list = [inp_new.split(" ")]
    inp_stats = preprocess(inp_list)
    stats_df = pd.DataFrame.from_dict(to_sound_list(inp_stats))
    discrete = encoder.transform(stats_df[["sym", "prev", "next",
                                           "next2", "nextBi", "nextTri",
                                           "prevBi", "curBi", "prevTri",
                                           "curTri", "curQuat"]])
    numeric = coo_matrix(stats_df[["beg", "end", "len",
                                   "vow", "vel", "logpos",
                                   "mult", "relBeg", "relEnd",
                                   "logBeg", "logEnd", "add",
                                   "prevVowel", "prevVelar",
                                   "nextVowel", "nextVelar"]].to_numpy())
    features = hstack([numeric, discrete])
    labels = model.predict(features)
    reusable_line = ""
    output_list = []
    for ind in range(labels.shape[0]):
        line = stats_df.loc[ind, ["sym", "next", "nextBi"]]
        reusable_line += line["sym"]
        if labels[ind] == 1:
            reusable_line += ">"
        if "$" in line["nextBi"]:
            reusable_line += line["next"]
            output_list.append(reusable_line)
            reusable_line = ""
    if len(output_list) == 1 or len(delimiters) == 0:
        return (init or "") + output_list[0] + (after or "")
# WORKAROUND
    if len(output_list) == len(delimiters) + 1:
        delimiters += [""]
    elif len(output_list) > len(delimiters) + 1:
        raise Exception("inconsistent number of words and delimiters")
# WORKAROUND
    output_list[-1] += delimiters[-1]
    output_string = ""
    for index in range(len(delimiters))[:-1]:
        output_string += output_list[index]
        output_string += delimiters[index]
    output_string += output_list[-1]
    output_string = (init or "") + output_string + (after or "")
    return output_string

if __name__ == '__main__':
    arg = docopt(__doc__, version='Kiche parser 0.1', options_first=False)
    if ( 
        not arg["--output"].endswith(".tsv")
        or not arg["--train"].endswith(".tsv")
        or not arg["--predict"].endswith(".tsv")):
        raise ValueError("all file names should end with .tsv")
    dat = pd.read_csv(arg["--train"], sep="\t", encoding="UTF-8", header=None)
    test = pd.read_csv(arg["--predict"], sep="\t", encoding="UTF-8", header=None)
    words_1 = tokenizer(dat.iloc[:,0])
    test_1 = tokenizer(test.iloc[:,0])
    sym_stats = preprocess(words_1)
    test_stats = preprocess(test_1)
    words2 = [i.split(" ") for i in dat.iloc[:,1]]
    for ind, sent in enumerate(words2):
        sent_process(sent, ind, sym_stats)
    neues = pd.DataFrame.from_dict(to_sound_list(sym_stats))
    all_tst = pd.DataFrame.from_dict(to_sound_list(test_stats)) 
    oneh = OneHotEncoder()
    temp = neues.iloc[:,:-1]
    common = pd.concat([temp, all_tst], axis=0)
    oneh.fit(common[["sym", "prev", "next", \
                     "next2", "nextBi", "nextTri", \
                     "prevBi", "curBi", "prevTri", \
                     "curTri", "curQuat"]])

    trans = oneh.transform(neues[["sym", "prev", "next", \
                                  "next2", "nextBi", "nextTri", \
                                  "prevBi", "curBi", "prevTri", \
                                  "curTri", "curQuat"]])

    tr_test = oneh.transform(all_tst[["sym", "prev", "next", \
                                      "next2", "nextBi", "nextTri", \
                                      "prevBi", "curBi", "prevTri", \
                                      "curTri", "curQuat"]])

    coo1 = coo_matrix(neues[["beg", "end", "len",
                             "vow", "vel", "logpos",
                             "mult", "relBeg", "relEnd",
                             "logBeg", "logEnd", "add", 
                             "prevVowel", "prevVelar",
                             "nextVowel", "nextVelar"]].to_numpy())

    coo2 = coo_matrix(all_tst[["beg", "end", "len",
                               "vow", "vel", "logpos",
                               "mult", "relBeg", "relEnd",
                               "logBeg", "logEnd", "add",
                               "prevVowel", "prevVelar",
                               "nextVowel", "nextVelar"]].to_numpy())

    alles = hstack([coo1, trans])
    tst_matrix = hstack([coo2, tr_test])
    y = neues["bound"].to_numpy(dtype=np.dtype(int))
    svm = LinearSVC(dual=False)
    svm.fit(alles, y)
    reserve = test.copy()
    for index, line in enumerate(reserve[0]):
        try:
            reserve.iloc[index, 1] = process_string(line, oneh, svm)     
        except:
            reserve.iloc[index, 1] = reserve.iloc[index, 0]
            print(f"failed to replace index {index}")
            print(reserve.iloc[index, 0] + "\n")
    reserve.to_csv(arg["--output"], sep="\t", header=False, index=False)        