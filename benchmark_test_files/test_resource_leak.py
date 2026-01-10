
f = open("test.txt")
data = f.read()
# Missing f.close()

with open("good.txt") as f2:
    data2 = f2.read()
