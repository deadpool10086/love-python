principal = 1000 # 初始金额
rate = 0.05 # 利率
numyears = 5 # 年数
year = 1
while year <= numyears:
    principal = principal * (1 + rate)
    #print(year, principal)
    #print("%3d %0.2f" % (year,principal))
    #print(format(year,"3d"),format(principal,"0.2f"))
    print("{0:3d} {1:0.2f}".format(year,principal))
    year += 1
