import numpy as np

matris1 = np.array([[1, 2], [3,4]])
matris2 = np.array([[5, 6], [7,8]])
carpim = np.dot(matris1, matris2)  #Skaler çarpım, Dot product
for eleman in range(len(matris1)):
    print(matris1[eleman], matris2[eleman], carpim[eleman])
print(matris1 * matris2)

