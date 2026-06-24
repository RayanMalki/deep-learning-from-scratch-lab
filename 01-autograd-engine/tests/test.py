from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b
d = c + Value(10.0)

d.backward()

print(a)
print(b)
print(c)
print(d)
