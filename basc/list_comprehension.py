students = [['lili',76],['qweqw',93],['weer',89],['qwe',86]]
avg = sum([x[1] for x in students]) / len(students)
print(avg)
sum = sum([i for i in range(1, 7) if 6 % i == 0])
print(sum)

# def f(a):
#     return a[1]
# students.sort(key = f, reverse = True)

students.sort(key = lambda x: x[1], reverse = True) #匿名函数

print(students)
