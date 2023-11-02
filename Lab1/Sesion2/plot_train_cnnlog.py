"""
Plot a Given a training log file.
"""
import csv
import matplotlib.pyplot as plt


def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies_t = []
        accuracies_v = []
        top_5_accuracies = []
        cnn_benchmark = []  # random results
        for epoch, acc, loss, val_acc, val_loss, in reader:
            accuracies_t.append(float(acc))
            accuracies_v.append(float(val_acc))
            cnn_benchmark.append(0.2)  # random

        plt.figure(figsize=(10,6))
        plt.plot(accuracies_t, label='acc_train')
        plt.plot(accuracies_v, label='acc_val')
        plt.plot(cnn_benchmark, label='cnn_benchmark')
        plt.legend()
        plt.savefig('inception-training-1698765901.0493932.png')
        plt.show()
        


if __name__ == '__main__':
    
    training_log = 'data/logs/inception-training-1698765901.0493932.log'
    main(training_log)
