from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.epoch = 0
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.epochs.append(self.epoch)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.accuracies.append(logs.get("accuracy"))
        self.val_accuracies.append(logs.get("val_accuracy"))
        self.epoch += 1
        
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5), sharex = True)
        
        clear_output(wait=True)
        ax1.plot(self.epochs, self.losses, label="loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.epochs, self.accuracies, label="accuracy")
        ax2.plot(self.epochs, self.val_accuracies, label="val_accuracy")
        ax2.legend()
        
        plt.show();