class Person(object):
    address = 'Earth'
    def __init__(self, name):
        self.name = name

p1 = Person('Bob')
p2 = Person('Alice')

print ('Person.address = ' + Person.address)

p1.address = 'China'
print ('p1.address = ' + p1.address)

print ('Person.address = ' + Person.address)
print ('p2.address = ' + p2.address)
del p1.address
print (p1.address)
