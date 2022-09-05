import math

def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
def process(xn,x,S):
    xnc1 = 0.5*(xn + S/xn)
    return xnc1

if __name__ == "__main__":
    inputt = float(input("nhap so: "))
    x = isqrt(inputt)
    xx = float(input("nhap x: "))
    print("x0: ",x,inputt)
    for i in range(1,xx):
        process()
    