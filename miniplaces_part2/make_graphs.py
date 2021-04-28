import matplotlib.pyplot as plt 

#top5error = 1-top5acc
lr = [1, 0.1, 0.01, 0.001, 0.0001]
lrResultTrain = [1-0.5252, 1-0.94083, 1-0.8241200000000001, 1-0.6089300000000001, 1-0.33636000000000005]
lrResultVal = [1-0.49870000000000003, 1-0.7069000000000001, 1-0.6818000000000001, 1-0.5770000000000001, 1-0.3362]

optim = ["SGD", "Adam", "Adagrad"]
optimResultTrain = [1-0.6089300000000001, 1-0.9553300000000001, 1-0.7536200000000001]
optimResultVal = [1-0.5770000000000001, 1-0.7218, 1-0.6607000000000001]

model = ["Resnet", "Alexnet", "VGG", "Densenet"]
modelResultTrain = [1-0.6089300000000001, 1-0.39319000000000004, 1-0.7046100000000001, 1-0.7924800000000001]
modelResultVal = [1-0.5770000000000001, 1-0.3915, 1-0.628, 1-0.7173]

plt.title("Learning Rate Error")
plt.plot(lr, lrResultVal)
plt.plot(lr, lrResultTrain)
plt.legend(["Val", "Train"])
plt.show()

plt.title("Optimizer Error")
plt.bar(optim, optimResultVal)
plt.bar(optim, optimResultTrain)
plt.legend(["Val", "Train"])
plt.show()

plt.title("Model Type Error")
plt.bar(model, modelResultVal)
plt.bar(model, modelResultTrain)
plt.legend(["Val", "Train"])
plt.show()


