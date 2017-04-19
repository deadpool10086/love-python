import json

class Students(object):
    def read(self):
        return r'["Tim", "Bob", "Alice"]'

s = Students()

print (json.load(s))  #json.load只调用s.read()只要存在read函数就行
