import os

filename = ('test_data/classical')
data = [1,2,3]


with open(filename, 'a') as f:
    f.write(str(data))
    f.write('\n\n')