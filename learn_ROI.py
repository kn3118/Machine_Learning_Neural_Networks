import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(2)
rn.seed(2)
tf.set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot

class ROI_Detector():

    def __init__(self, hyperparameters):
        self.hparams = hyperparameters

    def build(self):
        model = Sequential()
        model.add(Dense(self.hparams['neurons'][0], kernel_initializer=self.hparams['weight_initializer'][0], input_dim=self.hparams['input_dim'], activation=self.hparams['activation'][0]))
        model.add(Dropout(self.hparams['dropout_rate'][0], seed = 2))
        for i in range(1, len(self.hparams['activation'])):
            model.add(Dense(self.hparams['neurons'][i], activation=self.hparams['activation'][i], kernel_initializer=self.hparams['weight_initializer'][i]))
            model.add(Dropout(self.hparams['dropout_rate'][i], seed = 2))

        model.add(Dense(self.hparams['num_classes'], activation='softmax'))
        model.summary()
        self.model = model

    def train(self, x_train, y_train, x_val, y_val, weights, cb):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.hparams['optimizer'](lr = self.hparams['learning_rate'] ),
                           metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train,
                                      batch_size = self.hparams['batch_size'],
                                      epochs = self.hparams['epochs'],
                                      verbose = 1,
                                      validation_data = (x_val, y_val),
                                      class_weight = weights,
                                      callbacks = cb)
    
    def evaluate_architecture(self, x_test, y_test):

        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')

        pred_label = self.model.predict(x_test)
        predictions = []
        for i in range(len(pred_label)):
            max_row = np.max(pred_label[i])
            predictions.append([1 if x == max_row else 0 for x in pred_label[i]])
        print(classification_report(y_test, np.array(predictions)))
        print(confusion_matrix(y_test.argmax(axis=1), np.array(predictions).argmax(axis=1)))

        score = self.model.evaluate(x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy: %.2f%%' % (score[1]*100))

    def print_history(self):
        pyplot.plot(self.history.history['acc'], label='training')
        pyplot.plot(self.history.history['val_acc'], label='validation')
        pyplot.legend()
        pyplot.xlabel('# of epochs')
        pyplot.ylabel('Accuracy')
#        pyplot.savefig("Final2_acc.png")
        pyplot.show()
        pyplot.plot(self.history.history['loss'], label='training')
        pyplot.plot(self.history.history['val_loss'], label='validation')
        pyplot.legend()
        pyplot.xlabel('# of epochs')
        pyplot.ylabel('Loss')
#        pyplot.savefig("Final2_loss.png")
        pyplot.show()

    def save_model(self, filepath):
        self.model.save(filepath)
    
    def train_final(self, x_train, y_train, weights, cb):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.hparams['optimizer'](lr = self.hparams['learning_rate'] ),
                           metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train,
                                      batch_size = self.hparams['batch_size'],
                                      epochs = self.hparams['epochs'],
                                      verbose = 1,
                                      class_weight = weights,
                                      callbacks = cb)

def get_base_hyperparameters():
        
        return {
            'input_dim':        3, #Not a hyperparameter but here to keep inputs in one place
            'batch_size':       64,
            'num_classes':      4, #Not a hyperparameter but here to keep inputs in one place
            'epochs':           50,
            'neurons':          [3],
            'activation':       ['relu'],
            'learning_rate':    0.01,
            'dropout_rate':     [0],
            'optimizer':        SGD,
            'weight_initializer': ['he_uniform']
        }


def get_final_hyperparameters():
        
        return {
            'input_dim':        3, #Not a hyperparameter but here to keep inputs in one place
            'batch_size':       32,
            'num_classes':      4, #Not a hyperparameter but here to keep inputs in one place
            'epochs':           100,
            'neurons':          [8, 18, 18],
            'activation':       ['relu', 'relu','relu'],
            'learning_rate':    0.001,
            'dropout_rate':     [0, 0, 0.25],
            'optimizer':        Adam,
            'weight_initializer': ['he_uniform', 'he_uniform', 'he_uniform' ]
        }

def load_saved_model(filepath):
    model = load_model(filepath)
    return model

def predict_hidden(dataset):
    model = load_saved_model('final_Q3_model.h5')
    X = dataset[:, : 3 ]
    X = X.astype('float32')
    pred_label = model.predict(X)
    predictions = []
    for i in range(len(pred_label)):
        max_row = np.max(pred_label[i])
        predictions.append([1 if x == max_row else 0 for x in pred_label[i]])
    y = dataset[:, 3:  ]
    
    return np.array(predictions)

def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    X = dataset[:, : 3 ]
    y = dataset[:, 3 : ]
    
    #Splitting the data into Train, Validation and Test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.25, random_state = 2)
    
    #Generating weights for each class so as to counter the imbalanced dataset
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    weights = dict(enumerate(class_weights))

    ##############################################
    # Training and Evaluation
    ##############################################

#    Building, running and evaluating the base model
    ROI = ROI_Detector(get_base_hyperparameters())
    ROI.build()
    ROI.train(x_train, y_train, x_val, y_val, weights, cb=None)
    ROI.evaluate_architecture(x_val, y_val)
    ROI.print_history()
 
#    Building, running and evaluating the final model
    ROI = ROI_Detector(get_final_hyperparameters())
    ROI.build()
    cb = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    ROI.train(x_train, y_train, x_val, y_val, weights, cb)
    ROI.evaluate_architecture(x_val, y_val)
    ROI.print_history()
    ROI.train(x_train_val, y_train_val, x_test, y_test, weights, cb)
    ROI.evaluate_architecture(x_test, y_test)
    
#    Retraining final model parameters on entire dataset and saving it
    ROI = ROI_Detector(get_final_hyperparameters())
    ROI.build()
    cb = [EarlyStopping(monitor='loss', patience=10, verbose=1)]
    ROI.train_final(X, y, weights, cb)
    #filepath = "final_Q3_model.h5"
    #ROI.save_model(filepath)
    
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

if __name__ == "__main__":
    # main()
    dataset = np.loadtxt("ROI_dataset.dat")
    predictions = predict_hidden(dataset)
    print("Predictions:")
    print(predictions)