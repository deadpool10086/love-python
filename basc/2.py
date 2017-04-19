f = open("foo.txt") #读取一个文件对象
line = f.readline() #调用文件的readline()方法
# while line:
#     #print line, #python 2 后面跟‘，’将忽略换行
#     print(line,end='')
#     line = f.readline()
for line in open("foo.txt"):
    line = line.strip() #去掉开头和结束的空格个回车
    line = line.title #首字母大写 其余小写
    print(line)
f.close()
