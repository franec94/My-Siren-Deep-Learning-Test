import numpy as np
import matplotlib.pyplot as plt

def show_training_loss_graph(total_steps, history):

    train_loss_h = [] # val_loss_h = []
    train_loss_h = [h for h in history['train_loss']] # val_loss_h = [h for h in history['val_loss']]

    plt.title("Train Lss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,total_steps+1), train_loss_h, label="train_loss") # plt.plot(range(1,total_steps+1), val_acc_h, label="val_loss")
    # plt.ylim((0,1.))
    plt.xticks(np.arange(1, total_steps+1, 1.0))
    plt.legend()
    plt.show()
    pass
    