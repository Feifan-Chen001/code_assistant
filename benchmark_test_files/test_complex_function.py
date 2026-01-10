
def complex_func(a, b, c, d, e, f):
    if a > b:
        if c > d:
            if e > f:
                return a + c + e
            else:
                return a + c + f
        else:
            if e > f:
                return a + d + e
            else:
                return a + d + f
    else:
        if c > d:
            if e > f:
                return b + c + e
            else:
                return b + c + f
        else:
            if e > f:
                return b + d + e
            else:
                return b + d + f
