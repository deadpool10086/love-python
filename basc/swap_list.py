def swap(list, a, b):
    temp = list[a]
    list[a] = list [b]
    list[b] = temp

x = [10, 20, 30]
print(x)
swap(x, 0, 1)
print(x)
