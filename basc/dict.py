my_dict = {'John':1234, 'Bob':5645, 'Mike':54687}
print(my_dict)
my_dict['tom'] = 878
print(my_dict)
print(len(my_dict))
print('tom' in my_dict)
for key in my_dict:
    print(key)
print(my_dict.items())

s = "sdfsdfsdf"
lst = [0]*26
for i in s:
    lst[ord(i) - 97] += 1
print(lst)
d = {}
for i in s:
    if i in d:
        d[i] += 1
    else:
        d[i] = 1
print(d)
