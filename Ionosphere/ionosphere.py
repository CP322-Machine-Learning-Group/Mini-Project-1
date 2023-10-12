from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt  
import numpy as np
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
X = ionosphere.data.features 
y = ionosphere.data.targets 
  
# metadata 
#print(ionosphere.metadata) 
  
# variable information 
#print(ionosphere.variables) 

