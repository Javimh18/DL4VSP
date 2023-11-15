"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt


def main(crnn_training_log, cnn_training_log):
    with open(crnn_training_log) as fin_crnn, open(cnn_training_log) as fin_cnn:
        crnn_reader = csv.reader(fin_crnn)
        next(crnn_reader, None)  # skip the header
        cnn_reader = csv.reader(fin_cnn)
        next(cnn_reader, None) # skip the header

        random_fixed = []
        crnn_accuracies_t = []
        crnn_accuracies_v = []
        cnn_accuracies_t = []
        cnn_accuracies_v = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        #        for epoch,acc,loss,top_k_categorical_accuracy,val_acc,val_loss,val_top_k_categorical_accuracy in reader:
        for epoch, acc, loss, val_acc, val_loss in crnn_reader:
            crnn_accuracies_t.append(float(acc))
            crnn_accuracies_v.append(float(val_acc))
            random_fixed.append(0.2)
        
        for epoch, acc, loss, val_acc, val_loss in cnn_reader:
            cnn_accuracies_t.append(float(acc))
            cnn_accuracies_v.append(float(val_acc))
            
        plt.plot(crnn_accuracies_t, label='CRNN accuracy', color='lightskyblue')
        plt.plot(crnn_accuracies_v, label='CRNN val_accuracy', color='deepskyblue')
        plt.plot(cnn_accuracies_t, label='CNN accuracy', color='salmon')
        plt.plot(cnn_accuracies_v, label='CNN val_accuracy', color='red')
        plt.plot(random_fixed, label='Random vs Fixed Be', color='green')

        plt.legend()
        plt.title("Comparison CNN vs CRNN for seq = 5")
        plt.savefig('comp_cnn_crnn_5-seq.png')
        print("Figure saved.")
        plt.show()


if __name__ == '__main__':
    crnn_training_log = 'data/logs/lstm-5-training-1699524074.177566.log'
    cnn_training_log = 'data/logs/inception-training-1699639028.1059806.log'

    main(crnn_training_log, cnn_training_log)
