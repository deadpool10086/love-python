import time
from functools import reduce #python3 要导入
def performance(unit):
    def performance_decorator(f):
        def win(*args, **kw):
            t1 = time.time()
            r = f(*args, **kw)
            t2 = time.time()
            t = (t2 - t1) * 1000 if unit == 'ms' else (t2 - t1)
            print ("call %s() in %f %s" % (f.__name__,t,unit))
            return r
        return win
    return performance_decorator
@performance('ms')  #装饰器的过程实际上就是函数返回
def factorial(n):
    return reduce(lambda x,y: x*y, range(1, n+1))

print( factorial(10))
print(factorial.__name__)
input()
