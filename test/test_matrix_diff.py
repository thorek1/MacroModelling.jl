import sympy as sp

X = sp.MatrixSymbol('X', 7, 21)
Y = sp.MatrixSymbol('Y', 7, 21)
A = sp.MatrixSymbol('A', 7, 7)
B = sp.MatrixSymbol('B', 21, 21)

# Define the function
F = X + Y - A * Y * B

sp.diff(F,X)
sp.diff(F,A)

