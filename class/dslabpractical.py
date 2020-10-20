import math
def poisson(m,x):
	f=1
	for i in range(1,x+1):
		f*=i
	return ((m**x)*math.e**(-m))/(f)
ans1=0
for i in range(6):
	ans1+=poisson(10,i)
print('Probability that the number of calls will be maximum 5:',ans1)
ans2=0
for i in range(31):
	ans2+=poisson(30,i)
print('Probability that the number of calls over a 3 hour period will exceed 30:',1-ans2)