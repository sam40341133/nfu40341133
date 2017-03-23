class Set:

 def __init__(self, values=None):
    self.dict = {} # each instance of Set has its own dict property
 # which is what we'll use to track memberships
    if values is not None:
        for value in values:
            self.add(value)
 def __repr__(self):
    return "Set: " + str(self.dict.keys())
 def add(self, value):
    self.dict[value] = True

 def contains(self, value):
    return value in self.dict
 def remove(self, value):
    del self.dict[value]

s = Set([1,2,3])
s.add(4)
print s.contains(4) # True
s.remove(3)
print s.contains(3)

def exp(base, power):
    return base ** power
def two_to_the(power):
    return exp(2, power)
A = exp(2,3)
print(A)

from functools import partial
two_to_the = partial(exp, 2) # is now a function of one variable
print two_to_the(3)

def double(x):
    return 2 * x
def multiply(x, y):
    return x * y
xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs]
twice_xs = map(double, xs)
print(twice_xs)
list_doubler = partial(map, double) # *function* that doubles a list
twice_xs = list_doubler(xs)
print(twice_xs)
products = map(multiply, [1, 2], [4, 5]) # [1 * 4, 2 * 5] = [4, 10]
def is_even(x):
 """True if x is even, False if x is odd"""
 return x % 2 == 0

x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens = filter(is_even, xs) # same as above
list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs)
x_product = reduce(multiply, xs) # = 1 * 2 * 3 * 4 = 24
list_product = partial(reduce, multiply) # *function* that reduces a list
x_product = list_product(xs)


def doubler(f):
    def g(x):
        return 2 * f(x)
        return g
def f1(x):
    return x + 1
g = doubler(f1)
print g(3)
print g(-1)
def f2(x, y):
    return x + y
g = doubler(f2)
print g(1, 2)
def magic(*args, **kwargs):
    print "unnamed args:", args
    print "keyword args:", kwargs
magic(1, 2, key="word", key2="word2")
def other_way_magic(x, y, z):
    return x + y + z
x_y_list = [1, 2]
z_dict = { "z" : 3 }
print other_way_magic(*x_y_list, **z_dict)