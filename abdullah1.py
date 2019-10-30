import sklearn.datasets
boston_data = sklearn.datasets.load_boston()

print(dir(boston_data))

#print(boston_data.DESCR)

print(type(boston_data.data))

from numpy import shape
print(shape(boston_data.data))

print(boston_data.feature_names)

print(type(boston_data.target))

print(shape(boston_data.target))

import pandas as pd
df = pd.DataFrame(boston_data.data, columns=(boston_data.feature_names))
print(df.head())


df['Price'] = boston_data.target
print(df.describe())

# from numpy.random import randint
# numbers = randint(1, 100, 50)  

# import matplotlib.pyplot as plt
# plt.hist(numbers)
# plt.show()

import matplotlib.pyplot as plt

# plt.hist(df['Price'])
# plt.xlabel('price ($1000)')
# plt.ylabel('count')
# plt.show()

# for feature_name in boston_data.feature_names:
#     plt.scatter(df[feature_name], boston_data.target)
#     plt.ylabel('price', size=15)
#     plt.xlabel(feature_name, size=15)
#     plt.savefig(feature_name+".png")
#     plt.show()

# df['Price'] = boston_data.target
# print(df.corr())
import seaborn as sns
# sns.heatmap(df.corr())
# plt.show()

# sns.jointplot(df['Price'], df['LSTAT'], kind='hex')
# plt.show()

# from numpy import polyfit, polyval
# lstat = df['LSTAT']
# prices = df ['Price']
# p= polyfit(lstat, prices, 1)
# plt.plot(lstat, prices, 'o')
# plt.plot(sorted(lstat), polyval(p, sorted(lstat)), '-')
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['LSTAT'], boston_data.target)