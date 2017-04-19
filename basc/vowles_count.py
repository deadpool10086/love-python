def vowles_count(s):
    count = 0
    for c in s:
        if c in "aeiouAEIOU":
            count += 1
    return count
print(vowles_count("Hello world"))
