# 例子 1
import logging
case1 = [9, 8, 7, 3, 2, 2]
case1.append(1)
# print(case1.append(1))
# None


# 例子 2：去除数组里的偶数，因为数组是可变的，所以remove后index会改。
case2 = [9, 8, 8, 3, 3, 1]
# for i in case2:
#     if i % 2 == 0:
#         case2.remove(i)

# print(case2)
# [9, 8, 3, 3, 1]

case22 = []
for i in case2:
    if i % 2 == 1:
        case22.append(i)
# print(case22)  # [9, 3, 3, 1]

case222 = [x for x in case2 if x % 2]   # 使用列表推导式更简单高效
# print(case222)  # [9, 3, 3, 1]


# 例子 3：
case3 = 'a' 'b'     # 等于'a'+'b'
# print(case3)        # ab


# 例子 4：
flag = True
x, y = (10, 12) if flag else (None, None)
# print(x, y)


# 例子 5：
data = [1, 4, 5, 7, 9]
# for i in range(len(data)):
#     if data[i] % 2:
#         data[i] = 0
for idx, val in enumerate(data):
    if val % 2:
        data[idx] = 0
# print(data)


# 666666 字典排序
nums = [-1, -10, 0, 9, 5]
new_nums = sorted(nums, reverse=True)
# print(new_nums)

people = [
    {'name': 'jia', 'age': 18},
    {'name': 'yi', 'age': 60},
    {'name': 'bing', 'age': 20},
]
new_peo = sorted(people, key=lambda x: x['age'])
# print(new_peo)


# 77777 去重复set, 交集 (set1 & set2) 并集 (set1 | set2) 差集(set1 - set2) 补集(set1 ^ set2)
dup = [1, 1, 2, 3, 3, 4, 5, 5, 7]
single = set(dup)
# print(single)     #{1, 2, 3, 4, 5, 7}


# 8888 字典取值，如果没有key 直接赋值
person = {'name': 'Tom', 'age': 18}
# uid = person['uid'] # KeyError报错
uid = person.get('uid', 8888)
id = person.setdefault('id', '001')
# print(uid)
# print(id)
# print(person)


# 9999 字符串拼接 .join()
strs = ['Hi', 'my', 'friend']
greet = ' '.join(strs) + '!'
# print(greet)


# 1010 合并字典(无视key是否重复)
human1 = {'name': 'Abby', 'age': 18}
human2 = {'name': 'Emma', 'uid': 1234}
human3 = {'uid': 9999, 'gender': 'male'}
humans = {**human1, **human2, **human3}
# print(humans)   # {'name': 'Emma', 'age': 18, 'uid': 9999, 'gender': 'male'}


# 1111 if中的条件存在多个判断，可以把条件放到数组里 然后用if in
color = 'white'
# if color == 'blue' or color == 'yellow' or color == 'white':
#     print('cool!')

# colors = ['blue', 'yellow', 'white']
# if color in colors:
#     print('cool!')


# 1212 列表推导式
def case_12():
    numbers = [0, 1, 2, 3]
    d = {i: i * i for i in numbers}
    l = [i * i for i in numbers]
    s = {i * i for i in numbers}
    t = (i * i for i in range(4))


symbols = '!@#$%^&*'
codes = []
for symbol in symbols:
    codes.append(ord(symbol))
# print(codes)

codes_2 = [ord(symbol) for symbol in symbols if ord(symbol) > 50]
# print(codes_2)


out = []
for i in range(11):
    if i % 2 == 0:
        out.append(i)
# print(out)
odds = [num for num in range(11) if num % 2 == 0]
# print(odds)


my_nums = list(range(1, 11))
out = []
for i in my_nums:
    if i > 5:
        out.append(0)
    else:
        out.append(i)
out = [0 if i > 5 else i for i in my_nums]
# print(out)


my_dict = {i: i ** 2 for i in range(1, 6)}
new_dict = {k + 10: v ** 2 for (k, v) in my_dict.items()}
# print(new_dict)


# 元组 ### 不可变的列表；没有字段名的记录; 拆包unpack
a, b = divmod(10, 8)
# print(a, b)
a, b = b, a
# print(a, b)

test = (10, 8)
quotient, remainder = divmod(*test)

x, y, *rest = range(10)
# 0 1 [2,3,...,9]
x, *middle, y = range(10)
# 0, [1...8], 9
*first, x, y = range(10)
# [0...7], 8, 9
