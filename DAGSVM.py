from preprocess import load_data, data_clustering, one_versus_one
import itertools
import numpy as np
from SVM import SVM
from plotting import plot_confusion_matrix

def DAG_decide(combination, evaluate_matrix):
    ''' evaluate_matrix is (1000, 45) '''
    prediction = []
    for row in evaluate_matrix:
        prediction_dict = {}
        head, end = 0, 9
        for i in range(len(combination)):
            prediction_dict[combination[i]] = row[i]
        while head != end:
            pair = (head, end)
            if prediction_dict[pair] < 0:
                head += 1
            else:
                end -= 1
        prediction.append(end)
    return np.array(prediction)

X_train, Y_train, X_test, Y_test = load_data()
num_test = len(Y_test)

print('Clustering data...')
X_clustered_np = data_clustering(X_train, Y_train)

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

combination = list(itertools.combinations(numbers, 2))

evaluate_matrix = []
prediction = []

for pair in combination:
    X_positive, X_negative, Y_positive, Y_negative = one_versus_one(X_clustered_np, pair)

    training_data = np.vstack((X_positive, X_negative))
    testing_data = np.hstack((Y_positive, Y_negative))

    clf = SVM(C=10)
    clf.fit(training_data, testing_data)

    Y_evaluate = clf.evaluate(X_test)
    evaluate_matrix.append(Y_evaluate)

#(45, 1000)
evaluate_matrix = np.array(evaluate_matrix)
evaluate_matrix = evaluate_matrix.T

prediction = DAG_decide(combination, evaluate_matrix)

# Creating Confusion Matrix
confusion_matrix = np.zeros((len(numbers), len(numbers)))

for i in range(len(prediction)):
    confusion_matrix[prediction[i]][Y_test[i]] += 1

# Plotting
plot_confusion_matrix(confusion_matrix, 'Confusion Matrix for DAGSVM', 'DAGSVM.png')

correct = np.trace(confusion_matrix)
accuracy = (correct / num_test) * 100

print("%d out of %d predictions correct" % (correct, num_test))
print("The accuracy in percentage is  ")
print(accuracy)