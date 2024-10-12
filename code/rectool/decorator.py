import time
from functools import wraps


# 装饰器函数, 打印函数开始，和结束的时间
def time_this(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print(f"{f.__name__} cost time: {time.time()-start}")
        return result
    return wrapper