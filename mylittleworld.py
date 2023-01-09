#2023/01/06
#Gradient Descent Practice
#liner regretion practice


import matplotlib.pyplot as plt

xlist = []
ylist = []
x1 = -3
x2 = 3


def diviing_list(x1,x2,num,list):

  if x2 < x1:
    temp = x2
    x2 = x1
    x1 = temp
  appendnum = x1
  for i in range(0,num):
    appendnum = (x2-x1)/num + appendnum
    list.append(appendnum)
    
    
diviing_list(x1,x2,xlist)
print(xlist)


"""
import matplotlib.pyplot as graph


while(1):
    try:
        cnt = int(input("how many data will you put it? : "))
        break
    except:
        print("the input made an error! try again. (please type only the number)")
print("okay good, type in the data now")
x = []
y = []
for i in range(0,cnt):
    x.append(float(input("type x : ")))
    y.append(float(input("type y : ")))
print("okay done let me cacluate the (almost) perfect line!")

#grad

repeat = 1000#how many times are you going to do it?
rate = 0.01 #leaning rate
m = 1
b = 0
for j in range(0,repeat):

  sum = 0
  for i in range(0,cnt):
    sum = -(y[i]-(m*x[i]+b))#cost partial derivative by b
    MS_error = sum/cnt
    b = b - MS_error*rate
    

  sum = 0
  for i in range(0,cnt):
    sum = -x[i]*(y[i]-(m*x[i]+b))#cost partial derivative by m
    m = m - sum/cnt*rate
    
    

print("m(slope) is %d",m)
print("b(y intercept) is %d",b)
graph.scatter(x,y,label = "input dots")#dots
graph.axline((1, m+b), slope=m, color="black", linestyle=(0, (5, 5)),label ='ai graph')#line
graph.legend()
graph.show()
"""


print("lets see if its going to update or not")