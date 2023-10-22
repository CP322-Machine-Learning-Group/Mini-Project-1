from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt  
import numpy as np
from regression import LogReg
from regression import KNN

# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
X = ionosphere.data.features 
y = ionosphere.data.targets 
  
# metadata 
#print(ionosphere.metadata) 
  
# variable information 
#print(ionosphere.variables)
#print(ionosphere.data.features) 
#print(ionosphere.data.targets)

#for col in range(y.shape[0]):


Y=[]
j=2
i=0
sizeOfY =y.shape[0]
sizeOfX =X.shape[1]
y=np.array(y)
while i<len(y):
    if y[i]=='g':
        Y.append(1)
        i+=1
    else:
        Y.append(0)
        i+=1

    

y = np.array(Y) #(4601, 1)
X = np.array(X) #(4601, 57)

lr = LogReg(learning_rate=0.1, num_epochs=100)
losses = lr.fit(X, y)

accuracy = lr.evaluate_acc(X, y)
print('Accuracy: ', accuracy)

knn = KNN(k=3)
accuracy = KNN.k_fold_cross_validation(knn, X, y, k=5)
print("5-Fold Cross-Validation Accuracy:", accuracy)



plt.figure()
plt.plot(losses)

plt.grid()
plt.show()


