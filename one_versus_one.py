from preprocess import load_data, data_clustering, one_versus_one
import itertools
import numpy as np
from SVM import SVM
from plotting import plot_confusion_matrix

def label_prediction(Y_predict, pair):
    for i in range(len(Y_predict)):
        if  Y_predict[i] == 1:
            Y_predict[i] = pair[0]
        elif Y_predict[i] == -1:
            Y_predict[i] = pair[1]
    return Y_predict

X_train, Y_train, X_test, Y_test = load_data()
num_test = len(Y_test)

X_clustered_np = data_clustering(X_train, Y_train)

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

combination  = list(itertools.combinations(numbers, 2))

predict_matrix = []
prediction = []

for pair in combination:
    X_positive, X_negative, Y_positive, Y_negative = one_versus_one(X_clustered_np, pair)

    training_data = np.vstack((X_positive, X_negative))
    testing_data = np.hstack((Y_positive, Y_negative))

    clf = SVM(C=10)
    clf.fit(training_data, testing_data)

    Y_predict = label_prediction(clf.predict(X_test), pair)
    predict_matrix.append(Y_predict)

predict_matrix = np.array(predict_matrix).astype(int)
predict_matrix = predict_matrix.T

for row in range(predict_matrix.shape[0]):
    counts = np.bincount(predict_matrix[row])
    prediction.append(np.argmax(counts))

prediction = np.array(prediction)

# Creating Confusion Matrix
confusion_matrix = np.zeros((len(numbers), len(numbers)))

for i in range(len(prediction)):
    confusion_matrix[prediction[i]][Y_test[i]] += 1

# Plotting
plot_confusion_matrix(confusion_matrix, 'Confusion Matrix for One-Versus-One SVM', 'one-versus-one.png')

correct = np.trace(confusion_matrix)
accuracy = (correct / num_test) * 100

print("%d out of %d predictions correct" % (correct, num_test))
print("The accuracy in percentage is  ")
print(accuracy)