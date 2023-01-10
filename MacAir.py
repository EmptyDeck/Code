
#23/01/05
#liner regretion practice

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
    x.append(int(input("type x : ")))
    y.append(int(input("type y : ")))
print("okay done let me cacluate the (almost) perfect line!")


#avr
total_y = 0
total_x = 0
for i in x:
    total_x = total_x + i
avr_x = total_x/cnt
for i in y:
    total_y = total_y + i
avr_y = total_y/cnt

sum_xy = 0#this is sum of(xi-avr_x)(yi-(avr_y)
sum_xx = 0#this is sum of xi*avr_x
for i in range(0,cnt):
    sum_xy = sum_xy + (avr_x-x[i])*(avr_y-y[i])
    sum_xx = sum_xx + ((avr_x)-(x[i]))**2


m = sum_xy/sum_xx
b = avr_y-m*avr_x

print("the slope is %d",m)
print("and the y intercept is %d",b)

#graph.grid(color = "gary",alpha = 5,linestyle = "- -")
graph.scatter(x,y,label = "input dots")
graph.axline((1, m+b), slope=m, color="black", linestyle=(0, (5, 5)),label ='ai graph')
graph.legend()
graph.show()
print("Done!")