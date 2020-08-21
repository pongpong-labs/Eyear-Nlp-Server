from flask import Flask, request, jsonify
from gensim.models.word2vec import Word2Vec
from eunjeon import Mecab
from korean_romanizer import *
import jellyfish
from pykospacing import spacing

app = Flask(__name__)

model = Word2Vec.load('psychology.model')

tagger = Mecab()

text_list = []
with open('analysis_token.txt', 'r', encoding='utf-8') as f:
    for line in f:
        text_list.append(line.rstrip())

except_words = []
with open('except_word.txt', 'r', encoding='utf-8') as f:
    except_words = []
    for line in f:
        except_words.append(line.rstrip())

josa_eomi = []
with open('josaeomi.txt", 'r', encoding='utf-8') as f:
    for line in f:
        josa_eomi.append(line.rstrip())


# MAIN
def find_error_word():  # 오류가 있는지 확인
    global error_word
    error_word = []  # 오류단어를 저장
    for i in jamak_nouns:
        if i not in text_list:
            error_word.append(i)
    return error_word


def comb_error_word():
    global err_comb
    global err_word
    err_word = []
    err_comb = []
    for i in range(len(error_word)):
        for j in range(len(jamak)):
            if error_word[i] == jamak[j]:
                comb1 = []
                comb2 = []

                k = 0
                while jamak[j + k] not in josa_eomi:
                    real_err_word1 = jamak[j + k]
                    comb1.append(real_err_word1)
                    k += 1

                k = 1
                while jamak[j - k] not in josa_eomi:
                    real_err_word2 = jamak[j - k]
                    comb2.insert(0, real_err_word2)
                    k += 1

                err_comb.append(comb2 + comb1)

    for i in range(len(err_comb)):
        err_comb[i] = ''.join(err_comb[i])
    for v in err_comb:
        if v not in err_word:
            err_word.append(v)
    return err_word


def nearby_error_word():  # 오류 단어 앞뒤의 3단어 뽑기
    global check_nouns
    check_nouns = []  # 오류단어 앞뒤의 단어 저장
    if err_word == []:
        return []
    else:
        for i in range(len(err_word)):
            for j in range(len(line_space)):
                if err_word[i] == line_space[j] or err_word[i] in tagger.nouns(line_space[j]) or error_word[
                    i] in tagger.nouns(line_space[j]):
                    check_nouns_list = []
                    if j == len(line_space) - 1:
                        try:
                            li1 = tagger.nouns(line_space[j - 1])
                            li2 = tagger.nouns(line_space[j - 2])
                            li3 = tagger.nouns(line_space[j - 3])
                            li4 = tagger.nouns(line_space[j - 4])
                            li5 = tagger.nouns(line_space[j - 5])
                            li6 = tagger.nouns(line_space[j - 6])
                            check_nouns_list.extend(li1 + li2 + li3 + li4 + li5 + li6)
                        except:
                            pass

                    elif j == len(line_space) - 2:
                        try:
                            li1 = tagger.nouns(line_space[j - 1])
                            li2 = tagger.nouns(line_space[j + 1])
                            li3 = tagger.nouns(line_space[j - 2])
                            li4 = tagger.nouns(line_space[j - 3])
                            li5 = tagger.nouns(line_space[j - 4])
                            li6 = tagger.nouns(line_space[j - 5])
                            check_nouns_list.extend(li1 + li2 + li3 + li4 + li5 + li6)
                        except:
                            pass

                    else:
                        try:
                            li1 = tagger.nouns(line_space[j - 1])
                            li2 = tagger.nouns(line_space[j + 1])
                            li3 = tagger.nouns(line_space[j - 2])
                            li4 = tagger.nouns(line_space[j + 2])
                            li5 = tagger.nouns(line_space[j - 3])
                            li6 = tagger.nouns(line_space[j - 4])
                            check_nouns_list.extend(li1 + li2 + li3 + li4 + li5 + li6)
                        except:
                            pass

                    check_nouns.append(check_nouns_list)
                    if check_nouns[0] == []:
                        return []
    return check_nouns


def check_word_list():  # 오류 단어 근처의 단어에 대해 word2vec으로 학습한 연관성 높은 단어의 리스트를 출력
    global word_list
    word_list = []
    global model_result
    if check_nouns == []:
        return []

    else:
        for i in range(len(check_nouns)):
            list_result = []
            for j in check_nouns[i]:
                try:
                    model_result = model.wv.most_similar(j, topn=100)
                    for k in model_result:
                        if k[0] not in except_words:
                            list_result.append(k[0])
                except:
                    pass
            word_list.append(list_result)

            if list_result == []:
                return []

    return word_list


def romanizing():  # word_list의 한글 발음을 로마자로 변환
    global pronounce
    pronounce = []  # word_list의 발음을 저장.
    if word_list == []:
        return []
    else:
        for i in range(len(word_list)):
            pronounce_list = []
            for j in range(len(word_list[i])):
                try:
                    a = Romanizer(word_list[i][j])
                    pronounce_list.append(a.romanize())
                except:
                    pass
            pronounce.append(pronounce_list)
    return pronounce


def similarity():  # error word와 word list 단어의 발음 유사도 측정
    global probability
    probability = []

    if pronounce == []:
        return []

    else:
        for e in range(len(err_word)):
            a = Romanizer(err_word[e]).romanize()
            prob = []
            prob1 = []
            prob2 = []
            prob3 = []
            try:
                for j in range(len(pronounce[e])):
                    prob1.append(jellyfish.jaro_winkler_similarity(a, pronounce[e][j]))
                    prob2.append(jellyfish.jaro_similarity(a, pronounce[e][j]))
                    prob3.append(1 - (jellyfish.levenshtein_distance(a, pronounce[e][j])) / 7)
                    prob.append(prob3[j] + (prob1[j] + prob2[j]))
                probability.append(prob)
            except:
                pass
        if pronounce[0] == []:
            return []
    return probability


def word_change():  # 오류단어를 교체
    global correct_word
    global line_space
    correct_word = []

    if probability == []:
        return line_space

    else:
        change_word = []
        try:
            for i in range(len(probability)):
                err_word_index = probability[i].index(max(probability[i]))
                correct_word.append(word_list[i][err_word_index])

            for a in range(len(err_word)):
                for b in range(len(line_space)):
                    if err_word[a] == line_space[b]:
                        line_space[b] = correct_word[a]
        except:
            pass
    return line_space


def word_change2():
    global line_space

    if probability == []:
        return line_space

    else:
        try:
            line_space = word_change()
            line_nnn = spacing(''.join(line_space))
            line_nnn = tagger.morphs(line_nnn)
            for a in range(len(err_word)):
                for b in range(len(line_nnn)):
                    if err_word[a] == line_nnn[b]:
                        line_nnn[b] = correct_word[a]
                        line_space = line_nnn
        except:
            pass

    return line_space


def process(line):
    global jamak_nouns
    global jamak
    global line_for_space
    global line_space
    global change_word2

    jamak_nn = tagger.nouns(line)  # 자막에 나오는 의미있는 명사를 찾기
    line_for_space = line.split(' ')
    line_space = line_for_space.copy()

    for i in range(len(line_for_space)):
        if 2 * i < len(line_for_space):
            line_for_space.insert(2 * i + 1, 'space')

    for i in range(len(line_for_space)):
        line_for_space[i] = tagger.morphs(line_for_space[i])

    line_space_include = sum(line_for_space, [])
    jamak_nn = tagger.nouns(line)
    space_jamak = ''.join(line_space_include)

    jamak_nouns = []
    for w in jamak_nn:
        if w not in except_words:
            jamak_nouns.append(w)

    if tagger.morphs(space_jamak) != []:  # 자막문장을 품사별로 끊기
        jamak = tagger.morphs(space_jamak)

    error_word = find_error_word()
    err_word = comb_error_word()
    check_nouns = nearby_error_word()
    word_list = check_word_list()
    pronounce = romanizing()
    probability = similarity()
    change_word = word_change()
    change_word2 = word_change2()

    # 자막으로 전환

    final_line = ''.join(change_word2)

    result = spacing(final_line)

    return result


@app.route("/correctSubtitle", methods=['POST'])
def correctSubtitle():
    params = request.get_json()
    result = process(params['subtitle'])
    return {'result': result}


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=False)
