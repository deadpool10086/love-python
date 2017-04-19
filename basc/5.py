a = "hello world"
print(a)
b = a[4]
print(b)
e = a[3:8]  #s[i:j] 截取字符串s[k] i<=k<j
print(e)
x = "37"
y = "42"
z = x + y
print(z)
z = int(x) + int (y)
print(z)
x = 100
s = "The value of x is " + str(x) #与print 结果相同
print(s)
s = "The value of x is " + repr(x) #对象的精确表示
print(s)
s = "The value of x is " + format(x,"4d")
print(s)
# x = 3.4
# print("str(3.4) is"+str(x),"repr(3.4) is" + repr(3.4))
