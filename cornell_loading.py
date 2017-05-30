import pickle
import os.path
import operator
from numpy import array, transpose
from math import ceil
from config import Config as conf
from data_utility import *
import re


# bullshit_lines = ['L474', 'L24609', 'L239088', 'L283548', 'L303243', 'L535288', 'L535203',
#                   'L535163', 'L535148', 'L535133', 'L541062', 'L540986', 'L540485', 'L540483',
#                   'L540476', 'L540816', 'L619064', 'L50129', 'L78957']

def load_conversations(filename, bullshit_lines, script):
    f = open(filename, 'r')
    convID = 0
    conversations = {}

    for line in f:
        dialogue = line.strip().split(' +++$+++ ')

        lineIDs = dialogue[3][1:-1].strip().split(",")
        lineIDs = [line.strip()[1:-1] for line in lineIDs]

        for i in range(len(lineIDs)):
            if ((i + 1 < len(lineIDs)) and
                    (lineIDs[i] not in bullshit_lines) and
                    (lineIDs[i + 1] not in bullshit_lines)):
                d1 = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", script[lineIDs[i]])
                d2 = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", script[lineIDs[i+1]])
                print(d1)
                print(d2)
                dlg = [d1, d2]
                conversations[convID] = dlg
                convID = convID + 1
    f.close()
    return conversations, convID


def load_lines(filename):
    f = open(filename, 'r', encoding="ISO-8859-1")

    script = {}
    bullshit_lines = []

    for line in f:
        script_line = line.strip().split(' +++$+++ ')
        if len(script_line) < 5:
            bullshit_lines.append(script_line[0])
        else:
            script[script_line[0]] = script_line[4]

    print(bullshit_lines)
    f.close()
    return script, bullshit_lines

def dump_Tuples(filename, conversations, convID):
    f = open(filename, 'w')

    for i in range(convID):
        f.write("{}\t{}\n".format(conversations[i][0], conversations[i][1]))

    f.close()

def triples_to_tuples_check(input_filepath):

    f = open(input_filepath, 'r')
    count = 0
    count_equal_wzero = 0
    count_equal_nzero = 0
    for line in f:
        triples = line.strip().split('\t')

        for i in range(3):
            q1 = triples[i].count("``")
            q2 = triples[i].count("''")
            print("`` appears {} times and '' appears {} times".format(q1, q2))
            if q1 != q2:
                count = count + 1
            else:
                if q1 != 0:
                    count_equal_nzero = count_equal_nzero + 1
                count_equal_wzero = count_equal_wzero + 1



    f.close()
    print("Count of sentences with different number of `` and '': {}".format(count))
    print("Count of sentences with same number of `` and '' including 0 appearances: {}".format(count_equal_wzero))
    print("Count of sentences with same number of `` and '' EXCluding 0 appearances: {}".format(count_equal_nzero))

def tuples_check(filename):
    f = open(filename, 'r', encoding="ISO-8859-1")
    count = 0
    count_equal_wzero = 0
    count_equal_nzero = 0
    count_quotes_even = 0
    count_quotes_odd = 0
    for line in f:
        dialogue = line.strip().split("\t")
        for i in range(2):
            q1 = dialogue[i].count("``")
            q2 = dialogue[i].count("''")
            q3 = dialogue[i].count("\"")
            print("`` appears {} times and '' appears {} times and \" appears {} times".format(q1, q2, q3))
            if q1 != q2:
                count = count + 1
            else:
                if q1 != 0:
                    count_equal_nzero = count_equal_nzero + 1
                count_equal_wzero = count_equal_wzero + 1
            if q3 % 2 == 0:
                count_quotes_even = count_quotes_even + 1
            else:
                count_quotes_odd = count_quotes_odd + 1

    f.close()
    print("Count of sentences with different number of `` and '': {}".format(count))
    print("Count of sentences with same number of `` and '' including 0 appearances: {}".format(count_equal_wzero))
    print("Count of sentences with same number of `` and '' EXCluding 0 appearances: {}".format(count_equal_nzero))
    print("Count of sentences with EVEN number of \" is: {}".format(count_quotes_even))
    print("Count of sentences with ODD number of \" is: {}".format(count_quotes_odd))

def mainFunc():
    # script, bullshit_lines = load_lines(conf.CORNELL_lines_path)
    # conversations, convID = load_conversations(conf.CORNELL_conversations_path, bullshit_lines, script)
    # #print(conversations)
    # dump_Tuples(conf.CORNELL_TUPLES_PATH, conversations, convID)
    triples_to_tuples_check(TRAINING_FILEPATH)
    tuples_check(conf.CORNELL_TUPLES_PATH)



if __name__ == "__main__":
    mainFunc()
