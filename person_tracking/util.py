import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from config import Model, Data

n_classes = Model.N_CLASSES
n_epochs_hold = Model.LSTM_ARCH['n_epochs_hold']
n_epochs_decay = Model.LSTM_ARCH['n_epochs_decay']
epochs = Model.EPOCHS
LABELS = Data.LABELS
def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch[i] = _train[index]

    return batch

def getLRScheduler(optimizer):
    def lambdaRule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs_hold) / float(n_epochs_decay + 1)
        return lr_l

    schedular = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaRule)
    #schedular = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return schedular

def plot(x_arg, param_train, param_test, label, lr):
    plt.figure()
    plt.plot(x_arg, param_train, color='blue', label='train')
    plt.plot(x_arg, param_test, color='red', label='test')
    plt.legend()
    if (label == 'accuracy'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Training and Test Accuracy', fontsize=20)
        plt.savefig('Accuracy_' + str(epochs) + str(lr) + '.png')
        plt.show()
    elif (label == 'loss'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Test Loss', fontsize=20)
        plt.savefig('Loss_' + str(epochs) + str(lr) + '.png')
        plt.show()
    else:
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training loss and Test loss with learning rate', fontsize=20)
        plt.savefig('Loss_lr_' + str(epochs) + str(lr) + '.png')
        plt.show()

def evaluate(net, X_test, y_test, criterion):
    test_batch = len(X_test)
    net.eval()
    test_h = net.init_hidden(test_batch)
    inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test.flatten('F'))
    if (torch.cuda.is_available() ):
            inputs, targets = inputs.cuda(), targets.cuda()

    test_h = tuple([each.data for each in test_h])
    output = net(inputs.float(), test_h)
    test_loss = criterion(output, targets.long())
    top_p, top_class = output.topk(1, dim=1)
    targets = targets.view(*top_class.shape).long()
    equals = top_class == targets

    if (torch.cuda.is_available() ):
            top_class, targets = top_class.cpu(), targets.cpu()

    test_accuracy = torch.mean(equals.type(torch.FloatTensor))
    test_f1score = metrics.f1_score(top_class, targets, average='macro')


    print("Final loss is: {}".format(test_loss.item()))
    print("Final accuracy is: {}". format(test_accuracy))
    print("Final f1 score is: {}".format(test_f1score))

    confusion_matrix = metrics.confusion_matrix(top_class, targets)
    print("---------Confusion Matrix--------")
    print(confusion_matrix)
    normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    plotConfusionMatrix(normalized_confusion_matrix)

def plotConfusionMatrix(normalized_confusion_matrix):
    plt.figure()
    plt.imshow(
        normalized_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    plt.show()