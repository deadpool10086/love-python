f = open('names.txt','r')

def is_panlindrom(name):
    low = 0
    high = len(name) - 1
    while low < high:
        if name[low] != name[high]:
            return False
        else:
            low += 1
            high -= 1
    return True
def is_ascending(name):
    p = name[0]
    for c in name:
        if p > c:
            return False
        p = c
    return True

for line in f:
    line = line.strip()
    if is_panlindrom(line):
        print(line.title())
f.close()
