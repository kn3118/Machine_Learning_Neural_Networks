import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(2)
rn.seed(2)
tf.set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from nn_lib import Preprocessor

class COORD_Regressor():

    def __init__(self, hyperparameters):
        self.hparams = hyperparameters


    def build(self):
        model = Sequential()
        model.add(Dense(self.hparams['neurons'][0], kernel_initializer=self.hparams['weight_initializer'][0], input_dim=self.hparams['input_dim'], activation=self.hparams['activation'][0]))
        model.add(Dropout(self.hparams['dropout_rate'][0], seed = 2))
        for i in range(1, len(self.hparams['activation'])):
            model.add(Dense(self.hparams['neurons'][i], activation=self.hparams['activation'][i], kernel_initializer=self.hparams['weight_initializer'][i]))
            model.add(Dropout(self.hparams['dropout_rate'][i], seed = 2))

        model.add(Dense(3))
        model.summary()
        self.model = model

    
    def train(self, x_train, y_train, x_val, y_val, cb):
        self.model.compile(loss='mean_absolute_error',
                           optimizer=self.hparams['optimizer'](lr = self.hparams['learning_rate'] ),
                           metrics=['mean_squared_error'])
		
        self.history = self.model.fit(x_train, y_train,
                                      batch_size = self.hparams['batch_size'],
                                      epochs = self.hparams['epochs'],
                                      verbose = 1,
                                      validation_data = (x_val, y_val),
                                      callbacks = cb)
		
		
    def train_final(self, x_train, y_train, cb):
        self.model.compile(loss='mean_absolute_error',
                           optimizer=self.hparams['optimizer'](lr = self.hparams['learning_rate'] ),
                           metrics=['mse', 'accuracy'])
        self.history = self.model.fit(x_train, y_train,
                                      batch_size = self.hparams['batch_size'],
                                      epochs = self.hparams['epochs'],
                                      verbose = 1,
                                      callbacks = cb)	
	
    def evaluate_architecture(self, x_eval, y_eval):

        x_eval = x_eval.astype('float32')
        y_eval = y_eval.astype('float32')
        y_pred = self.model.predict(x_eval, verbose=0)
        
        error   = y_pred - y_eval
        x_error = (y_pred[:,0] - y_eval[:,0])
        y_error = (y_pred[:,1] - y_eval[:,1])
        z_error = (y_pred[:,2] - y_eval[:,2])
        print([np.min(np.abs(error)), np.max(np.abs(error))])
        
        RMSE = np.sqrt(np.mean(error**2))
        MAE  = np.mean(np.abs(error))
        print('RMSE: {:0.2f}'.format(RMSE))
        print('MAE:  {:0.2f}'.format(MAE))
        
        sns.distplot(np.sqrt(np.sum(error**2, axis=1)))
        plt.title('Total Resultant Error Distribution (per example): sqrt(dx^2 + dy^2 + dz^2)')
        plt.xlabel('Error [mm]')
        plt.show()
        
        sns.distplot(error.reshape(-1))
        plt.title('Error Distribution per regressed coordinate (dx/dy/dz)')
        plt.xlabel('Error [mm]')
        plt.show()        

    
    def print_history(self):
        pyplot.plot(np.sqrt(self.history.history['mean_squared_error']), label='RMSE')
        pyplot.plot(np.sqrt(self.history.history['val_mean_squared_error']), label='RMSE validation')
        pyplot.plot(self.history.history['loss'], label='MAE')
        pyplot.plot(self.history.history['val_loss'], label='MAE validation')
        #pyplot.plot(self.history.history['val_acc'], label='validation')
        pyplot.legend()
        pyplot.xlabel('# of epochs')
        pyplot.ylabel('Error [mm]')
        pyplot.savefig("Train history.png")
        pyplot.show()

    
    def save_model(self, filepath):
        self.model.save(filepath)


def get_base_hyperparameters():
        
        return {
            'input_dim':        3,
            'batch_size':       64,
            'epochs':           150,
            'neurons':          [6],
            'activation':       ['relu'],
            'learning_rate':    0.01,
            'dropout_rate':     [0],
            'optimizer':        SGD,
            'weight_initializer': ['he_uniform']
        }


def get_final_hyperparameters():
        
        return {
            'input_dim':        3,
            'batch_size':       64,
            'epochs':           150,
            'neurons':          [128]*4,
            'activation':       ['relu', 'sigmoid', 'tanh', 'relu'],
            'learning_rate':    0.001,
            'dropout_rate':     [0]*4,
            'optimizer':        Adam,
            'weight_initializer': ['he_uniform', 'glorot_uniform', 'glorot_uniform', 'he_uniform']
        }


def load_saved_model(filepath):
    model = load_model(filepath)
    return model


def predict_hidden(dataset):
    model = load_saved_model('final_Q2_model.h5')
    X = dataset[:, :3]
    X = X.astype('float32')
    predictions = model.predict(X)
    
    return np.array(predictions)


def main():
    dataset = np.loadtxt("FM_dataset.dat")
	
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
	
    X = dataset[:, :3]
    y = dataset[:, 3:]
    
    
    
    ratios = [0.6, 0.2, 0.2] # Split ratios [train, val, test]

    #Splitting the data into Train, Validation and Test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=ratios[1], random_state=2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=ratios[2]/(1-ratios[1]), random_state=2)
    
    # Pre-process data
#    pre = Preprocessor(x_train)
#    x_train = pre.apply(x_train)
#    x_test_pre = pre.apply(x_test)
#    x_val_pre = pre.apply(x_val)
    
#    #Building, running and evaluating the base model
#    MODEL = COORD_Regressor(get_base_hyperparameters())
#    MODEL.build()
#    MODEL.train(x_train_val, y_train_val, x_test, y_test, cb=None)
#    MODEL.evaluate_architecture(x_test, y_test)
#    MODEL.print_history()
#    MODEL.save_model("base_model.h5")
    
    #Building, running and evaluating the final model
    MODEL = COORD_Regressor(get_final_hyperparameters())
    MODEL.build()
    cb = [EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)]
    cb = []
    MODEL.train(x_train, y_train, x_val, y_val, cb)
    MODEL.evaluate_architecture(x_val, y_val)
    MODEL.print_history()
    MODEL.evaluate_architecture(x_test, y_test)
    
    #Retraining final model parameters on entire dataset and saving it
#    MODEL = COORD_Regressor(get_final_hyperparameters())
#    MODEL.build()
#    cb = [EarlyStopping(monitor='loss', patience=15, verbose=1)]
#    MODEL.train_final(X, y, cb)
#    MODEL.print_history()
#    MODEL.save_model("final_Q2_model.h5")
    
    #predictions = predict_hidden(dataset)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

if __name__ == "__main__":
#    main()
    dataset = np.loadtxt("FM_dataset.dat")
    predictions = predict_hidden(dataset)
    print(predictions)
    