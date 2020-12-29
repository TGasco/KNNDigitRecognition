# COMP219 Assignment 1
# By Thomas Gascoyne
# Student Number: 201419827
# Email: sgtgasco@liverpool.ac.uk
# Importing required modules
import joblib
import numpy as np
from sklearn import datasets, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Finds the Euclidean Distance between a train and test sample
def euclideanDistance(digit_a, digit_b, pixels):
    distance = 0
    for i in range(pixels):
        distance += (digit_a[i] - digit_b[i]) ** 2
    return np.sqrt(distance)


# Finds the k nearest neighbours
def findNearestNeighbours(Distances, k):
    kNN = np.empty(())
    numbers = np.empty(())
    for i in range(len(Distances)):
        if i % 2 == 0:
            kNN = np.append(kNN, [Distances[i]])
        else:
            numbers = np.append(numbers, [Distances[i]])
    indexes = np.argsort(kNN)

    newArr = np.empty(())
    for i in range(k):
        newArr = np.append(newArr, [numbers[indexes[i]]])
    return newArr


# Sorts and then classifies array
def classify(newArr):
    newArr = newArr.astype('int64')
    counts = np.bincount(newArr)
    classification = np.argmax(counts)
    return classification


# prints the accuracy report for the model
def accuracyReport(realValue, Prediction):
    np.set_printoptions(suppress=True)
    print("Classification report for classifier: \n %s\n%s\n"
          % (Prediction, metrics.classification_report(realValue[1:], Prediction[1:])))

    accuracy = accuracy_score(realValue[1:], Prediction[1:])
    error = 1 - accuracy
    print("Accuracy: ", accuracy)
    print("Train/Test Error: ", error)
    print()
    return accuracy


# Compares the predicted value against the real value
def queryAccuracy(realValue, prediction):
    correct = False
    if prediction != realValue:
        correct = True

    return print("Predicted digit: ", prediction,
                 "\nReal digit: ", realValue,
                 "\nMisclassified?: ", correct)


# Retrains both models, called when user changes the test size or k number
def retrainModels(x, y, k, pixels, test_size):
    print("Re-training f2 Model...")
    f2(x, y, k, test_size, True)
    print("Re-training f3 Model...")
    f3(k, x, y, pixels, test_size, True)
    return "Retrain complete, check f1 to see if the variables have updates correctly"


# Gives the details of the dataset used
def f1(x, y, pixels, test_size):
    print("Dataset for Handwriting recognition\n"
          "Digit Images \n --------------------------------\n", x,
          "\nDigit Labels \n --------------------------------\n", y,
          "\nDetails:",
          "\n-----------------------------",
          "\nNumber of samples: ", len(x),
          "\nNumber of classes: 10",
          "\nNumber of features per sample: ", pixels,
          "\nMin value for x: ", f'{np.amin(x):.0f}',
          "\nMax value for x: ", f'{np.amax(x):.0f}',
          "\nMin value for y: ", np.amin(y),
          "\nMax value for y: ", np.amax(y),
          "\nTrain test split: ", f'{100-(100*test_size):.0f}', "/", f'{(100*test_size):.0f}',
          "\nShape of x: ", x.shape,
          "\nShape of y: ", y.shape)

    return


# KNN algorithm to predict handwritten digits implemented using sklearn library functions
def f2(x, y, k, test_size, retrain):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    clf.fit(x_train, y_train)
    knnPredict = clf.predict(x_test)

    accuracyReport(y_test, knnPredict)

    # If, for whatever reason, you wish to re-dump the models, simply uncomment these
    if retrain == True:
        filename = 'f2Model.sav'
        joblib.dump([clf, knnPredict], filename)
    return


# Loads and implements the saved model for the f2 algorithm (see f2 for details)
def f2Model(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    clf, knnPredict = joblib.load("f2Model.sav")
    f2Accuracy = accuracyReport(y_test, knnPredict)
    return f2Accuracy


# Queries f2 algorithm
# Asks user to input an int to represent an index in the test set
# Finds the predicted and real value
def queryf2(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    testIndex = int(input("Enter an index from the test set: "))

    clf, knnPredict = joblib.load("f2Model.sav")
    prediction = knnPredict[testIndex]
    realValue = y_test[testIndex]

    return queryAccuracy(realValue, prediction)


# KNN algorithm to predict handwritten digits implemented using standard library (w/o sklearn)
def f3(k, x, y, pixels, test_size, retrain):
    train_x = x[:int((len(x) * (1 - test_size)))]
    lentrain_x = len(train_x)

    test_x = x[(lentrain_x + 1):len(x)]
    train_y = y[:int((len(y) * (1 - test_size)))]
    test_y = y[(len(train_y) + 1):len(y)]

    realValueArr = np.empty(())
    PredictionArr = np.empty(())

    for count in range(len(test_x) - 1):
        Distances = np.empty((lentrain_x, 0))
        for i in range(lentrain_x - 1):
            trainDistances = np.array([[euclideanDistance(test_x[count], train_x[i], pixels)], [train_y[i]]])
            Distances = np.append(Distances, [[trainDistances[0, :], trainDistances[1, :]]])

        SplitDistances = np.split(Distances, int(len(Distances)))
        kNN = findNearestNeighbours(SplitDistances, k)
        Prediction = classify(kNN)
        PredictionArr = np.append(PredictionArr, Prediction)
        realValue = test_y[count]
        realValueArr = np.append(realValueArr, realValue)

    realValueList = realValueArr.tolist()
    accuracyReport(realValueList, PredictionArr)
    # Saving
    if retrain == True:
        filename = "f3Model.sav"
        joblib.dump([PredictionArr, realValueArr], filename)
    return


# Loads and implements the saved model for the f3 algorithm (see f3 for details)
def f3Model():
    PredictionArr, realValueArr = joblib.load("f3Model.sav")
    realValueList = realValueArr.tolist()
    f2Accuracy = accuracyReport(realValueList, PredictionArr)
    print()
    return f2Accuracy


# Queries f3 algorithm
# Asks user to input an int to represent an index in the test set
# Finds the predicted and real value
def queryf3():
    testIndex = int(input("Enter an index from the test set: "))
    PredictionArr, realValueArr = joblib.load("f3Model.sav")
    prediction = PredictionArr[testIndex]
    realValue = realValueArr[testIndex]

    return queryAccuracy(round(realValue), round(prediction))


# Compares the accuracy and train/test errors for the f2 and f3 models
def f4(x, y, test_size):
    print("Accuracy for f2 model:\n")
    f2Accuracy = f2Model(x, y, test_size)
    print("Accuracy for f2 model:\n")
    f3Accuracy = f3Model()
    print("Accuracy for f2 model:\n", f2Accuracy)
    print("Train/test error for f2 model:\n", 1 - f2Accuracy)
    print("\nAccuracy for f3 model:\n", f3Accuracy)
    print("Train/test error for f3 model:\n", 1 - f3Accuracy)
    return


# main body of the program, this is where the functions are called
def main():
    digits = datasets.load_digits()
    pixels = digits.images.shape[1] * digits.images.shape[2]
    digits.images = digits.images.reshape(digits.images.shape[0], digits.images.shape[1] * digits.images.shape[2])
    test_size = 0.3
    x = digits.images
    y = digits.target
    k = int(np.sqrt(len(x)))

    while True:
        userInput = input("Press: \n"
                          "-------------------------\n"
                          "f1: details of dataset\n"
                          "f2: train sklearn implemented Knn algorithm\n"
                          "f3: train Knn algorithm manually\n"
                          "f4: compare train test errors for f2 and f3\n"
                          "f5a: Query the f2 model\n"
                          "f5b: Query the f3 model\n"
                          "op: Additional Options\n"
                          "f0: exit\n"
                          )
        # if-elif-else block takes the user input and executes the respective method
        if userInput == "f1":
            f1(x, y, pixels, test_size)
        elif userInput == "f2":
            f2Model(x, y, test_size)
        elif userInput == "f3":
            f3Model()
        elif userInput == "f4":
            f4(x, y, test_size)
        elif userInput == "f5a":
            queryf2(x, y, test_size)
        elif userInput == "f5b":
            queryf3()
        elif userInput == "f0":
            print("Thank you, Goodbye")
            break
        elif userInput == "op":
            acceptPrompt = """Please note: in order to set a new k, the models will have to be re-saved,
            which may take some time, if you wish to proceed, confirm with y,
            any other input will send you back to the main menu\n"""

            option = input("1. Change the test size (default = 0.3)\n"
                           "2. Change k (default = sqrt of the dataset)\n"
                           "3. train sklearn implemented Knn algorithm in Real Time\n"
                           "4. train Knn algorithm manually in Real Time\n"
                           "0. Reset to defaults\n")
            if option == "1":
                accept = input(acceptPrompt)
                if accept.casefold() == "y":
                    test_size = float(input("New test size: "))
                    print(retrainModels(x, y, k, pixels, test_size))
            elif option == "2":
                accept = input(acceptPrompt)
                if accept.casefold() == "y":
                    k = int(input("New k: "))
                    print(retrainModels(x, y, k, pixels, test_size))

            elif option == "3":
                f2(x, y, k, test_size, False)
            elif option == "4":
                f3(k, x, y, pixels, test_size, False)
            elif option == "0":
                accept = input(acceptPrompt)
                if accept.casefold() == "y":
                    test_size = 0.3
                    k = int(np.sqrt(len(x)))
                    print(retrainModels(x, y, k, pixels, test_size))
            else:
                print("Input error")

        else:
            print("Input error")

        print("\nEnter to continue...")
        input()
        print("\n")
    return


# executes program
main()
