
numbers = [1,2,3,4,5,6,7,8,9,10]
squared = list(map(lambda x:x**2,numbers))
print(f"squared list{squared}")
filters = list(filter(lambda x:x%2 == 0,numbers))
print(f"Even numbers: {filters}")