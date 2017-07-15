def str2byte(s):
    base='0123456789ABCDEF'
    i = 0
    s = s.upper()
    s1=''
    while i < len(s):
        c1=s[i]
        c2=s[i+1]
        i+=2
        b1=base.find(c1)
        b2=base.find(c2)
        s1+=chr((b1 << 4)+b2)
    return s1

s ='504354467B596F755F6172335F476F6F645F437261636B33527D'
print(str2byte(s))
