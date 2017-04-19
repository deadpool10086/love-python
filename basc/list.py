nums = []
for i in range(10):
    nums.append(float(input()))

s = 0
# for num in nums:
#     s += num
s = sum(nums)
avg = s / len(nums)

print(avg)
