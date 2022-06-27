import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_matrix (y_test, y_pred):

    # Creating the confusion matrix
    confusion_m = confusion_matrix(y_test, tf.round(y_pred))
    confusion_matrix_norm = confusion_m.astype("float") / confusion_m.sum(axis = 1)[:, np.newaxis]
    ''' The above line will normalize the matrix ''' 
    n_classes = confusion_m.shape[0] # To see number of classes (rows)

    # Prettifying it
    fig, ax = plt.subplots(figsize = (10, 10))

    # Matrix plot
    cax = ax.matshow(confusion_m, cmap = plt.cm.Blues) 
    fig.colorbar(cax)

    # Classes
    if confusion_m.shape[0] > 2:
      classes = True
    else :
      classes = False

    if classes:
      labels = classes # If we have multiclass then it would save the list to labels
    else:
      labels = np.arange(confusion_m.shape[0]) # else just the array of our binary label

    # Axes labels
    ax.set(title = "Confusion Matrix",
           xlabel = "Predicted label",
           ylabel = "True Label",
           xticks = np.arange(n_classes),
           yticks = np.arange(n_classes),
           xticklabels = labels,
           yticklabels = labels)

    # Thresold for different colors
    thresold = (confusion_m.max() + confusion_m.min()) / 2 # Average

    # Plot the text on each cell
    for i, j in itertools.product(range(confusion_m.shape[0]), range(confusion_m.shape[1])):
           ''' The above is just saying that put the rows as X and Columns as Y'''
           plt.text(j, i, f"{confusion_m[i, j]}({confusion_matrix_norm[i,j]*100: .1f}%)",
                        horizontalalignment = "center",
                        color = "white" if confusion_m[i,j] > thresold else "black",
                        size = 15)     
