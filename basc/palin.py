num = 12321
num_p = 0
num_t = num
while num != 0:
    num_p = num_p * 10 + num % 10
    num = num // 10
if num_t == num_p:
    print("ok")
else:
    print(num_t,num_p)
    print('no')
