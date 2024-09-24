import numpy as np

# Create array a
a = np.random.randint(0,5,(1, 5))  # a = [1, 2, 3, 4, 5]
print(a)

# Create array b with the same values of a along the third dimension
b = np.tile(a, (10, 10, 1))

c = np.random.randint(0,100,(10,10))

result = b*c[:,:,np.newaxis]

num1= np.random.randint(5)
print(f'the readom number is: {num1+1}')
print(b.shape)
print(b[3,5,:])
print(b[3,5,num1])
print(c.shape)
print(f'the number selected from 2D mat is: {c[3,5]}');
print(result[3,5,num1])
