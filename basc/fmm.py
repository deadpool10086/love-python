def load_dict(filename):
    word_dict = set()
    max_len = 1
    f = open(filename,'r',encoding='utf-8')
    for line in f:
        word = line.strip()
        word_dict.add(word)
        if len(word) > max_len:
            max_len = len(word)
    return max_len, word_dict
def fmm_word_seg(sent, max_len, word_dict):
    begin = 0
    words = []
    sent = sent
    while begin < len(sent):
        for end in range(begin + max_len, begin, -1):
            if sent[begin:end] in word_dict:
                words.append(sent[begin:end])
                break
        begin = end
    return words
max_len, word_dict = load_dict('lexicon.dic')
sent = input('input a sententce:')
words = fmm_word_seg(sent, max_len, word_dict)
for word in words:
    print(word)
