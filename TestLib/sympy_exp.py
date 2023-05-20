from sympy import *
import numpy as np
# create symbols n and k
k = symbols('k')
n = symbols('n')
# create a symbolic expression.
# NOTE: I have replace the upper limit n with infinity
expr1 = Sum((pi / 2)**(2 * k) / factorial(2 * k), (k, 0, oo))
expr1.doit()
print(expr1.doit())
# output: cosh(pi/2)

# If you wanted to compute the limit of the original expression:
expr2 = Sum((pi / 2)**(2 * k) / factorial(2 * k), (k, 0, n))
limit(expr2, n, oo)
# It throws an error!