num = 151

def is_prim(num):
    for i in range(2, num):
        if num % i == 0:
            return False
        else:
            return True
def is_pailn(num):
    num_p = 0
    num_t = num
    while num != 0:
        num_p = num_p * 10 + num % 10
        num = num // 10
    if num_t == num_p:
        return True
    else:
        return False
if is_prim(num) and is_pailn(num):
    print("yes it is")
else:
    print("no it's not")
