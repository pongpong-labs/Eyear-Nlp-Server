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


# MAIN
def find_error_word():  # 오류가 있는지 확인
    global error_word
    error_word = []
    for i in jamak_nouns:
        if i not in text_list:
            error_word.append(i)
    return error_word


def nearby_error_word():  # 오류 단어 앞뒤의 3단어 뽑기
    global check_nouns
    check_nouns = []  # 오류단어 앞뒤의 단어 저장
    if error_word == []:
        return []
    else:
        for i in range(len(error_word)):
            for j in range(len(jamak_nouns)):
                if error_word[i] == jamak_nouns[j]:
                    check_nouns_list = []
                    if j == len(jamak_nouns) - 1:
                        try:
                            check_nouns_list.append(jamak_nouns[j - 1])
                            check_nouns_list.append(jamak_nouns[j - 2])
                            check_nouns_list.append(jamak_nouns[j - 3])
                            check_nouns_list.append(jamak_nouns[j - 4])
                        except:
                            pass
                    elif j == len(jamak_nouns) - 2:
                        try:
                            check_nouns_list.append(jamak_nouns[j - 1])
                            check_nouns_list.append(jamak_nouns[j + 1])
                            check_nouns_list.append(jamak_nouns[j - 2])
                            check_nouns_list.append(jamak_nouns[j - 3])
                        except:
                            pass
                    else:
                        try:
                            check_nouns_list.append(jamak_nouns[j - 1])
                            check_nouns_list.append(jamak_nouns[j + 1])
                            check_nouns_list.append(jamak_nouns[j - 2])
                            check_nouns_list.append(jamak_nouns[j + 2])
                        except:
                            pass
                    check_nouns.append(check_nouns_list)
    return check_nouns


def check_word_list():  # 오류 단어 근처의 단어에 대해 연관성 높은 단어의 리스트를 저장
    global word_list
    global model_result
    word_list = []

    if check_nouns == []:
        return []

    else:
        for i in range(len(check_nouns)):
            list_result = []
            for j in check_nouns[i]:
                try:
                    model_result = model.wv.most_similar(j, topn=50)
                    for k in model_result:
                        if k[0] not in except_words:
                            list_result.append(k[0])
                except:
                    pass
            word_list.append(list_result)

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


def similarity():  # 유사도 측정
    global probability
    probability = []
    if pronounce == []:
        return []
    else:
        for e in range(len(error_word)):
            a = Romanizer(error_word[e]).romanize()
            prob = []
            prob1 = []
            prob2 = []
            prob3 = []
            for j in range(len(pronounce[e])):
                prob1.append(jellyfish.jaro_winkler_similarity(a, pronounce[e][j]))
                prob2.append(jellyfish.jaro_similarity(a, pronounce[e][j]))
                prob3.append(1 - (jellyfish.levenshtein_distance(a, pronounce[e][j])) / 21)
                prob.append(prob3[j] + (prob1[j] + prob2[j]) / 2)
            probability.append(prob)
    return probability


def word_change():  # 오류단어를 교체
    global correct_word
    correct_word = []
    if probability == []:
        return jamak
    else:
        change_word = []
        for i in range(len(probability)):
            err_word_index = probability[i].index(max(probability[i]))
            correct_word.append(word_list[i][err_word_index])

        for a in range(len(jamak)):
            for b in range(len(error_word)):
                if jamak[a] == error_word[b]:
                    jamak[a] = correct_word[b]
    return jamak


def process(line):
    global jamak_nouns
    global jamak
    jamak_nn = tagger.nouns(line)  # 자막에 나오는 의미있는 명사를 찾기

    jamak_nouns = []
    for w in jamak_nn:
        if w not in except_words:
            jamak_nouns.append(w)

    if tagger.morphs(line) != []:  # 자막문장을 품사별로 끊기
        jamak = tagger.morphs(line)

    error_word = find_error_word()
    check_nouns = nearby_error_word()
    word_list = check_word_list()
    pronounce = romanizing()
    probability = similarity()
    change_word = word_change()

    # 자막으로 전환

    jamak = ''.join(jamak)

    result = spacing(jamak)

    return result


@app.route("/correctSubtitle", methods=['POST'])
def correctSubtitle():
    params = request.get_json()
    result = process(params['subtitle'])
    return {'result': result}


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=False)
