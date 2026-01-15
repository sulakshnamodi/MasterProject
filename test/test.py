# What is literable and Iteration
# Iterable: a "container" you can loop over. E.g. lists, string, tuples, dictionaries, ranges
# Iteration: the process of looping
# 

# # variables (Initialising a variable)
# student_count = 1000 # integer
# rating = 4.99 #float
# is_publised = True # boolean
# course_name = "English Course" # string
# print(student_count)

# # String
# course = "Python Programming"
# print(course)
# print(len(course))
# print(course[0]) # to access the postion of an element
# print(course[0:5]) # to slice a string

# # formatted string
# first = "Sula"
# last = "Modi"

# full_name = first + " " + last

# f2 = f"{first} {last}"

# print(len(full_name), len(f2))


# # String Method
# new_course = "   Python Programming  "
# print(new_course)
# print(new_course.upper())
# print(new_course.lstrip() + "there is a space before this") # this remove the space before "python" on the left hand side
# print(new_course.find("Py"))
# print(new_course.replace("o", "Z"))
# print("py" in new_course)
# print("babu" not in new_course) 

# #number
# print(10 + 3)
# print(10 - 3)
# print(10 * 3)
# print(type(10 / 3)  , 10 / 3)
# print(type(10 // 3))
# print(10 % 3)
# print(10 ** 3) # its to the power of

# x = 10
# #x = x + 3
# x -= 3
# print(x)

# # For Loops
# for number in range(1, 5):
#     print("Attempt", number, number * ".")
# list = ['a','b','c']


# # Nested Loop
# for x in list:
#     for y in range(200):
#         for z in range(3):
#             print(f"({x} {y} {z})")

# Nested loop, 2nd example

# colorlist = ['red', 'blue', 'yellow', 'green']
# alphabetlist = ['a', 'b', 'c', 'd']


# for color1 in colorlist:
#     for i in range(3):
#         for alphabet in alphabetlist:
#             if alphabet not in ['a', 'c']:
#                 continue
#             print(f"{color1} {i} {alphabet}")

# # Nested loop, another example

# colourlist = ["white", "green", "orange"]

# for i in range(3):

#     for colour in colourlist:

#         if colour not in 'white':
#             print('Color is white, printing skipped')
#             continue # Go to next item in this loop

#         print(f"{i} {colour}")


# #List
# letters = ["a", "b", "c"]
# matrix = [[0, 1], [2, 3]]
# zeros = [0] * 5
# combined = letters + zeros
# numbers = list(range(20))
# chars = list("bubu dudu destiny")
# print(len(chars))

# #Accessing item in a list
# letters = ["a", "b", "c", "d"]
# letters[0] = "A"
# print(letters[0:3])
# print(letters[:3])
# print(letters[0:])

# numbers = list(range(20))
# print(numbers[::2])
# print(numbers[::-1])

# numbers = [1, 2, 3, 4, 4, 4, 10]
# first,*other, last = numbers
# print(first, last)
# print(other)

# #looping over list
# letters = ["a", "b", "c"]
# items = (0, "a")
# index, alphabet = items
# for index, alphabets in enumerate(letters):
#     print(index, alphabets)

#adding and removing item in a list
letters = ["a", "b", "c"]

#add
letters.append("d") #insert at the end 
letters.insert(1, "a*") #insert at selected position
print(letters)
# #remove
# letters.pop() # remove element at the end of the list
# letters.pop(1) #remove at the selected position
# letters.remove("c") #remove first occurance of c

# print(letters)

# alphabets = ["c", "b", "c", "c", "b", "c", "a"]

# for letter in alphabets:
#     if letter == "c":
#         print("there is c here, I skipped it")
#         continue
#     print(letter)

# # Finding Items
# letters = ["x", "y", "z", "x"]
# print(letters.count("x")) #count the number of time x appears
# if "z" in letters:
#     print(letters.index("z"))

# # Sorting list
# numbers = [3, 51, 2, 8, 6]
# #numbers.sort(reverse=True)
# print(sorted(numbers, reverse=True))
# print(numbers)

# # Sorting tuples
# items = [
#     ("product1", 10),
#     ("product2", 9),
#     ("product3", 12),

# ]

# #create a function to sort the item by its price
# def sort_item(item):
#     return item[1]

# items.sort(key=sort_item)
# print(items)
# items.sort(key=sort_item, reverse=True)
# print(items)

# # Zip Function
# list1 = [1, 2, 3]
# list2 = [10, 20, 30]

# print(list(zip("abc", list1, list2)))

# # tuples: it is a read only list, we cannot modify the element
# point = (1, 2)
# print(type(point))

# #converting a list to a tuple
# num = [4, 5, 6, 7]
# print(tuple(num))

# #WDID: zip() function in Python is used to combine multiple iterables (like lists, tuples, or strings) into a single iterator of tuples.


# # Dictionaries: collection of key value pairs. To map a key to a value.
# # example of phone number: Sarah (key) -> 56485231 (value)

# #this two line of code do the exact same thing, but second line of code is more easy
# #phone_num = {"Sarah": 5, "Adriana":7, "Babu":6}
# phone_num = dict(Sarah=5, Adriana=7, Babu=6)
# # we use the key ["Babu"] to access the value, not like in list where we use accessing position like this [0]
# phone_num["Babu"] = 10 # replace value 6 by 10
# phone_num["Dudu"] = 15 # add another value to the dictionary

# if "Sarah" in phone_num:
#     print(phone_num["Sarah"])
# print(phone_num)

# for key, value in phone_num.items():
#     print(key, value)

# print(phone_num.items())

# # WDID: .item() returns and display distionary's key and value as tuples




