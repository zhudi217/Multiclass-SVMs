import numpy as np
from preprocess import data_clustering, one_and_theRest, load_data
from SVM import SVM
from plotting import plot_confusion_matrix

X_train, Y_train, X_test, Y_test = load_data()
num_test = len(Y_test)

print("Clustering data...")
X_train_clustered = data_clustering(X_train, Y_train)

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

prediction_binary = []

for number in range(10):
    # For one-versus-all only
    print("Preparing data for one-versus-the-rest...")
    X_positive, X_negative, Y_positive, Y_negative = one_and_theRest(X_train_clustered, number)

    training_data = np.vstack((X_positive, X_negative))
    testing_data = np.hstack((Y_positive, Y_negative))

    print("[SVM] Start Training SVM...")
    clf = SVM(C=10)
    clf.fit(training_data, testing_data)

    Y_predict = clf.predict(X_test)
    prediction_binary.append(Y_predict)

prediction = np.argmax(np.array(prediction_binary), axis=0)

# Creating Confusion Matrix
confusion_matrix = np.zeros((len(numbers), len(numbers)))

for i in range(len(prediction)):
    confusion_matrix[prediction[i]][Y_test[i]] += 1

# Plotting
plot_confusion_matrix(confusion_matrix, 'Confusion Matrix for One-Versus-The-Rest SVM', 'one_versus_the_rest.png')

correct = np.trace(confusion_matrix)
accuracy = (correct / num_test) * 100

print("%d out of %d predictions correct" % (correct, num_test))
print("The accuracy in percentage is  ")
print(accuracy)