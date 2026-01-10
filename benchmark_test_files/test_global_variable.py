
count = 0

def increment():
    global count
    count += 1

def bad_usage():
    global unused_var
    pass
