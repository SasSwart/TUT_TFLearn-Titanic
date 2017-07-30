import numpy as np, tflearn

# Download Titanic Dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load Dataset, indicate that first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)


def pre_process(data, columns_to_ignore):
    # Sort Descending and ignore columns
    [[r.pop(id) for r in data] for id in sorted(columns_to_ignore, reverse=True)]
    for datum in data:
        datum[1] = 1. if data[1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore name and ticket columns
to_ignore = [1, 6]

pre_process(data, to_ignore)

net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
jaco = [3, 'Jaco Swart', 'male', 43, 1, 2, 'N/A', 5.0000]
durby = [3, 'Durby-Ann Swart', 'female', 40, 1, 2, 'N/A', 5.0000]
sas = [3, 'Sas Swart', 'male', 20, 0, 3, 'N/A', 5.0000]
tiaan = [3, 'Tiaan Swart', 'male', 18, 0, 3, 'N/A', 5.0000]
robert = [1, 'Robert Wood', 'male', 19, 2, 2, 'N/A', 100.0000]

dicaprio, winslet, jaco, durby, sas, tiaan, robert = pre_process([dicaprio, winslet, jaco, durby, sas, tiaan, robert], to_ignore)

prediction = model.predict([dicaprio, winslet, jaco, durby, sas, tiaan, robert])

print('Dicaprio\t:\t', prediction[0][1])
print('Winslet\t:\t', prediction[1][1])
print('Jaco\t:\t', prediction[2][1])
print('Durby\t:\t', prediction[3][1])
print('Sas\t\t:\t', prediction[4][1])
print('Tiaan\t:\t', prediction[5][1])
print('Robert\t:\t', prediction[6][1])
