import math

while True:
    a = int(input("nhap a (khac 0): "))
    if (a == 0):
        print("nhap sai, yc nhap lai!")
    else:
        break
b = int(input("nhap b: "))
c = int(input("nhap c: "))

delta = b**2 - 4*a*c
print("ket qua:\n")
if delta > 0:
    y1 = (- b - math.sqrt(delta)) / (2*a)
    y2 = (- b + math.sqrt(delta)) / (2*a)
    print("y1: ",y1)
    print("y2: ",y2)
elif (delta == 0):
    y = -b / 2*a
    print("y: ",y)
else:
    print("vo nghiem:))")