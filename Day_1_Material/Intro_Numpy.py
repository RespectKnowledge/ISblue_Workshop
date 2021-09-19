# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:58:27 2021

@author: Administrateur
"""

# Numpy
# Numpy is the backbone of the numerical and scientific computing stack in Python; 
# many of the libraries we'll cover in this course (SciPy, pandas, scikit-learn, etc.) 
# depend on it internally. 
# Numpy provides many data structures optimized  for efficient representation and manipulation 
# of different kinds of high-dimensional data, as well as an enormous range 
# of numerical tools that help us work with those structures. Because numpy 
# is so fundamental to scientific computing in Python,

# Importing numpy
# Recall that, in Python, the default namespace contains only a small number of built-in functions. To use any other functionality, we need to explicitly import it into our namespace.
# Let's do that for numpy:
# import numpy as np
     
# By convention, numpy is imported as np for brevity. 
# This is a general convention in Python; most widely-used packages 
# have standard abbreviations that everyone in the community uses. 
# While Python itself won't complain if you write, 
# say, import numpy as my_favorite_numerical_library, we strongly recommend sticking with the conventional abbreviations, as they make it easier for everyone else to understand what your code is doing at a glance.

# The NDArray (n-dimensional array)
# The core data structure in numpy is the n-dimensional array (or ndarray). 
# As the name suggests, an ndarray is an array with an arbitrary number 
# of dimensions. Unlike Python lists, numpy arrays are homogeneously typed—meaning, every element in the array has to have the same data type. You can have an array of floats, or an array of integers, but you can't have an array that mixes floats and integers (though numpy does have a structured array data type we won't cover here that provides a way of representing heterogeneous data).

# Creating NDArrays
# Like any other Python object, we need to initialize an ndarray before 
# we can do anything with it. Numpy provides us with several ways to create 
# new arrays. 

# Initializing an array from an existing list
# Let's start by constructing an array from existing data. 
# Assume we have some values already stored in a native Python 
# iterable object (typically, a list), and we want to convert 
# that object to an ndarray so that we can perform more 
# efficient numerical operations on it. In this case, 
# we can just pass the iterable object directly to np.ndarray().

import numpy as np
# A Python list of lists
my_list = [[1, 2, 5], [4, 1, 7]]

# Construct an array from the list
my_arr = np.array(my_list)

print(my_arr)

# Create a new 10 x 10 array
arr_2d = np.zeros((5, 10))

# Create a new 3-d array with dimensions 2, 4, 8
arr_3d = np.zeros((2, 4, 8))

# Returns an array of evenly-spaced values within a certain range
np.arange(4, 10)

# Generate normally distributed values drawn from a distribution with the specified shape.
# There are several other random number generating functions in the np.random module.
# Here we generate a 10 x 2 array of values sampled from a normal distribution with
# mean of 2 and sd of 4.
np.random.normal(2, 4, size=(10, 2))

# np.full() is like np.zeros() and np.ones(), but fills the array with the
# specified value instead of 0 or 1.
np.full((5, 5), 400)

my_list = [[1, 2, 5], [4, 1, 7]]

my_arr.shape
my_arr.ndim

print("dtype of my_arr:", my_arr.dtype)
print("dtype of arr_3d:", arr_3d.dtype)

np.zeros((5,), dtype=int)

idexing
arr = np.array([7, 1, 9, 14, 2, 8, 9, 6, 10, 4])

arr
arr[3]


arr[2:6] #6 is exlcuded

# create a 5 x 5 array of zeros
arr_2d = np.zeros((5, 5))
print(arr_2d)
# create a 3 x 3 array of ones.
# np.ones() is just like np.zeros(), but fills the matrix with ones
ones = np.ones((3, 3))
print(ones)
# replace a 3 x 3 sub-array of arr_2d with ones
arr_2d[:3, 2:] = ones

print(arr_2d)

x = np.arange(100)
print("x:", x)

# 1st argument is the array to be reshaped; 2nd is the target shape
np.reshape(x, (10, 10))

x = np.ones((5, 2))
print("x:\n", x)
print("\nx transposed:\n", np.transpose(x))

x = np.ones((5, 2))
print(x.T)

x = np.ones((10, 2))
y = np.ones((10, 2)) * 2
print(x.shape)
print(y.shape)
xy = np.concatenate([x, y], axis=0)

xy.shape

xy = np.concatenate([x , y], axis=1)

xy.shape

x = np.array([1, 2, 5, 9, 3, 1, 4, 5, 2, 9, 4])

np.split(x, [3, 2], axis=0)

# np.arange() builds an array with sequentially increasing values
x = np.arange(10)
print("x:", x)
### math operations
print("\nAdd 10 to x:", x + 10)
print("Divide x by 2:", x / 2)
print("x modulo 2:", x % 2)
print("x raised to the power of 3:", x ** 3)

y = np.array([1, 5, 48, -7, 12, 6, -4, 1.8, 9])

# Take the absolute value of each element
np.abs(y)

y = [4, 5, 1, 100, 75]

# Natural log of each element
np.log(y)

y = np.array([1, 5, 48, -7, 12, 6, -4, 1.8, 9])

print("Max of y:", np.max(y))
print("Min of y:", np.min(y))
print("Mean of y:", np.mean(y))
print("Median of y:", np.median(y))
print("Variance of y:", np.var(y))

### function along axis

x = np.arange(100)
x = np.reshape(x, (20, 5))

print("Means over the first axis (i.e., mean of each column):", np.mean(x, axis=0))
print("Means over the second axis (i.e., mean of each row):", np.mean(x, axis=1))

x = np.random.randint(0, 10, size=(2, 5))
y = np.random.randint(0, 10, size=(5, 2))

# Dot product
x.dot(y)

########### most powerful and useful function used in numpy array
#1. array concatenation
#2. reshaping
#3. seequzing diemnsion(reduce diemnsion)
#4. unseequze diemsnion(expand dimension)
#5. swap dimension

#numpy.squeeze
x = np.array([[[0], [1], [2]]])
print(x.shape)
print(x.ndim)
x_se=np.squeeze(x)

np.squeeze(x, axis=0).shape
np.squeeze(x, axis=1).shape
np.squeeze(x, axis=2).shape

x1 = np.array([[1234]])

print(x1.shape)

np.squeeze(x1)


x1=np.random.rand(10,20,1)
print(x1)
y=np.squeeze(x1,axis=2)
print(y.shape)

#numpy.expand_dims is equilant to unsqueeze array
x = np.array([1, 2])
print(x.shape)
y = np.expand_dims(x,axis=0) # horizontal expand x[np.newaxis, :]
print(y)

y1 = np.expand_dims(x,axis=1) #vertical exapand x[:, np.newaxis]
print(y1)
y = np.expand_dims(x, axis=(0, 1)) # expand in both axis
print(y.shape)

x1=np.random.rand(10,20,2)
print(x1)
y=np.expand_dims(x1,axis=0)
print(y.shape)

#numpy.reshape
a = np.zeros((10, 2))
print(a)
b = a.T
print(b)
c = b.view()
print(c)
a = np.arange(6).reshape((3, 2))
print(a)

a = np.array([[1,2,3], [4,5,6]])
print(a)
np.reshape(a, 6)

np.reshape(a, (3,-1)) ## the unspecified value is inferred to be 2

#numpy.swapaxes
x = np.array([[1,2,3]])
print(x)
print(x.shape)
np.swapaxes(x,0,1)
x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
print(x)
print(x.shape)
np.swapaxes(x,0,2)
x1=np.random.rand(10,20,2)
print(x1)
print(x1.shape)
x1_sa=np.swapaxes(x1,0,1)
print(x1_sa.shape)








