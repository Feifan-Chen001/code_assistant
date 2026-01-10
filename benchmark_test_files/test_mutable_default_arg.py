
def bad_func(items=[]):
    items.append(1)
    return items

def bad_func2(config={}):
    config["key"] = "value"
    return config
