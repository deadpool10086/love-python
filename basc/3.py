principal = 1000 # 初始金额
rate = 0.05 # 利率
numyears = 5 # 年数
year = 1

f = open("out.txt","w")
while year <= numyears:
    principal = principal * (1 + rate)
    # print("%3d %0.2f" % (year, principal), file=f)
    # #print >> f, "%3d %0.2f" % (year, principal) #py2 写法
    f.write("%3d %0.2f\n" % (year, principal))
    year += 1
f.close()
