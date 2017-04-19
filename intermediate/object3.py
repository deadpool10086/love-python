class Person(object):
    count = 0
    def __init__(self, name):
        self.name = name
        Person.count += 1

p1 = Person('Bob')
print (Person.count)

p2 = Person('Alice')
print (Person.count)

p3 = Person('Tim')
print (Person.count)

class Person(object):
    __count = 0
    def __init__(self, name):
        Person.__count = Person.__count + 1
        self.name = name
        print (Person.__count)

p1 = Person('Bob')
p2 = Person('Alice')

print (Person.__count)
