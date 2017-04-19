class Person(object):
    def __init__(self, name, score):
        self.name = name
        self.__score = score  # X.__私有属性  X._保护属性

p = Person('Bob', 59)

print( p.name)
print( p.__score)
