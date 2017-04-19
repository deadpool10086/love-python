words = ['sadfsd','asdgasdf','asdfsdf',\
    'asdfasdf','grt', 'sdfgsdfhshrt']
lis = []

words.sort(key = lambda x : len(x), reverse = True)
print(words)
# for word in words:
#     lis.append((len(word),word))
# lis.sort(reverse = True)
# res = []
#
# for length, word in lis:
#     res.append(word)
#
# print(res)
