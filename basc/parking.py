import random
def parking (low, high):  #随机在low 和 high 之间停 长度为1的车
    if high - low < 1:
        return 0
    else:
        x = random.uniform(low, high -1)
        return parking(low, x) + 1 + parking(x + 1, high)
sum = 0
for count in range(0, 100000):
    sum += parking(0, 5)
count += 1
print(sum / count)
