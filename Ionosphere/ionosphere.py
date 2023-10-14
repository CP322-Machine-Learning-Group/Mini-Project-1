from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt  
import numpy as np
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
x = ionosphere.data.features 
y = ionosphere.data.targets 
  
# metadata 
#print(ionosphere.metadata) 
  
# variable information 
#print(ionosphere.variables)
#print(ionosphere.data.features) 
print(ionosphere.data.targets)

#for col in range(y.shape[0]):
List=[]
j=2
i=0
sizeOfY =y.shape[0]
sizeOfX =x.shape[1]
y=np.array(y)

while i<len(y):
    if y[i]=='g':
        List.append('1')
        i+=1
    else:
        List.append('0')
        i+=1
print(List)
    


plt.figure()
plt.plot(List)

plt.grid()
plt.show()


