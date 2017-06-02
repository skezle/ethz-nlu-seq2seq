import pickle
import os.path
from config import Config as conf
from data_utility import *
import re

# bullshit_lines = ['L474', 'L24609', 'L239088', 'L283548', 'L303243', 'L535288', 'L535203',
#                   'L535163', 'L535148', 'L535133', 'L541062', 'L540986', 'L540485', 'L540483',
#                   'L540476', 'L540816', 'L619064', 'L50129', 'L78957']

def load_conversations(filename, bullshit_lines, script, characters_capslock, characters_firstupper):
    print("Loading dialogue turns from {} and applyting regular expressions..".format(filename))

    matching_quotationmarks = re.compile('\"(.*)\"')
    single_quotationmarks = re.compile('\"(.*)')

    space_number_space = re.compile(" \d+ ")
    space_number_tab = re.compile("\d+\t")
    number_space = re.compile("\d+ ")
    dash_number_dash = re.compile("-\d+-")
    dash_number = re.compile("-\d+")
    number_dash = re.compile("\d+-")
    number_st = re.compile("\d+st")
    number_nd = re.compile("\d+nd")
    number_rd = re.compile("\d+rd")
    number_th = re.compile("\d+th")
    number_am = re.compile("\d+am")
    number_pm = re.compile("\d+pm")
    number_m = re.compile("\d+m")
    number_mm = re.compile("\d+mm")
    number_a = re.compile("\d+a")
    number_b = re.compile("\d+b")
    number_c = re.compile("\d+c")
    number_d = re.compile("\d+d")
    number_g = re.compile("\d+g")
    number_w = re.compile("\d+w")
    number_k = re.compile("\d+k")
    space_number_s = re.compile("\d+s")
    number_dollar = re.compile("\d+\$")
    dollar_number = re.compile("\$\d+")
    number_plus = re.compile("\d+\+")
    space_number_slash = re.compile(" \d+/")
    slash_number_slash = re.compile("/\d+/")
    number_slash = re.compile("\d+/")
    slash_number = re.compile("/\d+")
    number_b_slash = re.compile("\d+b/")
    zeroh = re.compile("0h")
    dash_tagnumber = re.compile("-<number>")
    tagnumber_dash = re.compile("<number>-")
    tagperson = re.compile("< person >")
    f_number = re.compile("f\d+")
    e_number = re.compile("e\d+")
    d_number = re.compile("d\d+")
    c_number = re.compile("c\d+")
    b_number = re.compile("b\d+")

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

                d1 = script[lineIDs[i]]
                d2 = script[lineIDs[i + 1]]

                tokens_d1 = d1.strip().split(" ");  tokens_d2 = d2.strip().split(" ")
                new_tokens_d1 = [];                 new_tokens_d2 = []

                for i in range(len(tokens_d1)):
                    if ((tokens_d1[i] in characters_capslock) or (tokens_d1[i] in characters_firstupper)):
                        new_tokens_d1.append("<person>")
                    else:
                        new_tokens_d1.append(tokens_d1[i])

                for i in range(len(tokens_d2)):
                    if ((tokens_d2[i] in characters_capslock) or (tokens_d2[i] in characters_firstupper)):
                        new_tokens_d2.append("<person>")
                    else:
                        new_tokens_d2.append(tokens_d2[i])

                d1_string = new_tokens_d1[0];       d2_string = new_tokens_d2[0]
                for i in range(1, len(new_tokens_d1)):
                    d1_string = d1_string + " " + new_tokens_d1[i]
                for i in range(1, len(new_tokens_d2)):
                    d2_string = d2_string + " " + new_tokens_d2[i]

                d1 = d1_string.lower();     d2 = d2_string.lower()

                d1 = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", d1)
                d2 = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", d2)
                d1 = d1.replace("'", " ' ")
                d2 = d2.replace("'", " ' ")
                d1 = tagperson.sub("<person>", d1);     d2 = tagperson.sub("<person>", d2)
                # codes and passwords
                d1 = f_number.sub("f <number>", d1);    d2 = f_number.sub("f <number>", d2)
                d1 = e_number.sub("e <number>", d1);    d2 = e_number.sub("e <number>", d2)
                d1 = d_number.sub("d <number>", d1);    d2 = d_number.sub("d <number>", d2)
                d1 = c_number.sub("c <number>", d1);    d2 = c_number.sub("c <number>", d2)
                d1 = b_number.sub("b <number>", d1);    d2 = b_number.sub("b <number>", d2)
                # address and apartments
                d1 = number_b_slash.sub("<number> b / ", d1);    d2 = number_b_slash.sub("<number> b / ", d2)
                d1 = number_a.sub("<number> a", d1);    d2 = number_a.sub("<number> a", d2)
                d1 = number_b.sub("<number> b", d1);    d2 = number_b.sub("<number> b", d2)
                d1 = number_c.sub("<number> c", d1);    d2 = number_c.sub("<number> c", d2)
                d1 = number_d.sub("<number> d", d1);    d2 = number_d.sub("<number> d", d2)
                d1 = number_g.sub("<number> g", d1);    d2 = number_g.sub("<number> g", d2)
                # numbers with order numbers
                d1 = number_st.sub("<number> st", d1);  d2 = number_st.sub("<number> st", d2)
                d1 = number_nd.sub("<number> nd", d1);  d2 = number_nd.sub("<number> nd", d2)
                d1 = number_rd.sub("<number> rd", d1);  d2 = number_rd.sub("<number> rd", d2)
                d1 = number_th.sub("<number> th", d1);  d2 = number_th.sub("<number> th", d2)
                d1 = number_am.sub("<number> am", d1);  d2 = number_am.sub("<number> am", d2)
                d1 = number_pm.sub("<number> pm", d1);  d2 = number_pm.sub("<number> pm", d2)
                # units and measurements
                d1 = number_w.sub("<number> w", d1);        d2 = number_w.sub("<number> w", d2)
                d1 = number_k.sub("<number> k", d1);        d2 = number_k.sub("<number> k", d2)
                d1 = space_number_s.sub(" <number> s", d1); d2 = space_number_s.sub(" <number> s", d2)
                d1 = number_m.sub("<number> m", d1);        d2 = number_m.sub("<number> m", d2)
                d1 = number_mm.sub("<number> mm", d1);      d2 = number_mm.sub("<number> mm", d2)
                d1 = number_dollar.sub("<number> $", d1);   d2 = number_dollar.sub("<number> $", d2)
                d1 = dollar_number.sub("$ <number>", d1);   d2 = dollar_number.sub("$ <number>", d2)
                # number and dashes
                d1 = dash_number_dash.sub("- <number> -", d1);  d2 = dash_number_dash.sub("- <number> -", d2)
                d1 = dash_number_dash.sub("- <number> -", d1);  d2 = dash_number_dash.sub("- <number> -", d2)
                d1 = dash_number.sub("- <number>", d1);         d2 = dash_number.sub("- <number>", d2)
                d1 = number_dash.sub("<number> -", d1);         d2 = number_dash.sub("<number> -", d2)
                # rational numbers
                d1 = slash_number_slash.sub("/ <number> / ", d1);   d2 = slash_number_slash.sub("/ <number> / ", d2)
                d1 = space_number_slash.sub(" <number> / ", d1);    d2 = space_number_slash.sub(" <number> / ", d2)
                d1 = number_slash.sub("<number> / ", d1);           d2 = number_slash.sub("<number> / ", d2)
                d1 = slash_number.sub(" / <number>", d1);           d2 = slash_number.sub(" / <number>", d2)
                d1 = number_plus.sub("<number> + ", d1);            d2 = number_plus.sub("<number> + ", d2)
                # numbers and spaces
                d1 = space_number_space.sub(" <number> ", d1);  d2 = space_number_space.sub(" <number> ", d2)
                d1 = space_number_tab.sub(" <number>\t", d1);   d2 = space_number_tab.sub(" <number>\t", d2)
                d1 = space_number_space.sub(" <number> ", d1);  d2 = space_number_space.sub(" <number> ", d2)
                d1 = number_space.sub("<number> ", d1);         d2 = number_space.sub("<number> ", d2)
                # misc
                d1 = zeroh.sub("oh", d1);                   d2 = zeroh.sub("oh", d2)
                d1 = dash_tagnumber.sub("- <number>", d1);  d2 = dash_tagnumber.sub("- <number>", d2)
                d1 = tagnumber_dash.sub("<number> -", d1);  d2 = tagnumber_dash.sub("<number> -", d2)
                d1 = d_number.sub("d <number>", d1);        d2 = d_number.sub("d <number>", d2)
                # quotes
                d1 = single_quotationmarks.sub(r"``\1", matching_quotationmarks.sub(r"``\1''", d1))
                d2 = single_quotationmarks.sub(r"``\1", matching_quotationmarks.sub(r"``\1''", d2))

                dlg = [d1, d2]
                conversations[convID] = dlg
                convID = convID + 1
    f.close()
    return conversations, convID


def load_lines(filename):
    print("Loading script lines from: {}".format(filename))

    f = open(filename, 'r', encoding="ISO-8859-1")
    script = {}
    bullshit_lines = []
    characters_capslock = set()
    characters_firstupper = set()
    discarded_characters = set()

    for line in f:
        script_line = line.strip().split(' +++$+++ ')
        if len(script_line) < 5:
            bullshit_lines.append(script_line[0])
        else:
            script[script_line[0]] = script_line[4]
            if len(script_line[3].lower().strip()) > 1:
                s_cl = script_line[3].strip()
                s_fu = "".join(c if i == 0 else c.lower() for i, c in enumerate(script_line[3].strip()))
                #print("{}\t\t{}".format(s_cl, s_fu))
                characters_capslock.add(s_cl)
                characters_firstupper.add(s_fu)
            else:
                discarded_characters.add(script_line[3].lower().strip())

    print("Number of empty lines: {}".format(len(bullshit_lines)))
    print("Number of unique characters: {}".format(len(characters_capslock)))
    print("Number of one-letter names: {}".format(len(discarded_characters)))
    f.close()
    return script, bullshit_lines, characters_capslock, characters_firstupper, discarded_characters


def dump_Tuples(filename, conversations, convID):
    print("Dumping cleaned tuples on path {}".format(filename))
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
            #print("`` appears {} times and '' appears {} times".format(q1, q2))
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
            #print("`` appears {} times and '' appears {} times and \" appears {} times".format(q1, q2, q3))
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


def create_Cornell_tuples(lines_path, conversations_path, tumples_path):
    script, bullshit_lines, characters_capslock, character_firstupper, _ = load_lines(lines_path)
    conversations, convID = load_conversations(conversations_path, bullshit_lines, script, characters_capslock, character_firstupper)
    dump_Tuples(tumples_path, conversations, convID)


def mainFunc():
    create_Cornell_tuples(conf.CORNELL_lines_path, conf.CORNELL_conversations_path, conf.CORNELL_TUPLES_PATH)

if __name__ == "__main__":
    mainFunc()
