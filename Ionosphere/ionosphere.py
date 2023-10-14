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

while i< (sizeOfY):
    while j<(sizeOfX):
        value= x.iat[i,j]
        List.append(value)
        
        j+=1
    i+=1
    j=0


plt.figure()
plt.boxplot(List,patch_artist=True)

plt.grid()
plt.show()


