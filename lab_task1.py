                                       #  Group Number-10
                                       
                                       #  Group Members :
                                       #  Sumiaya Khatun, ID : 2212576141 
                                       #  Most. Rima Khatun, ID : 2112376130
                                       #  Mst. Sumyta Rahman, ID : 2212276116
                                       #  MD Fayed Hassan, ID : 2210176143 
                                       
                                       
#  Colab Link : https://colab.research.google.com/drive/16yQlG9NVJ_bP9KSVNDvPYzo1v4E_rlk_?usp=sharing

#import necessary library
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

#Define the polynomail function here
def polynomail(X):
  return 5*X**3-10*X**2-20*X+10

# Set Random Seed
np.random.seed(4)

# Generate data within range(-20 to 20) and data amount is 10000
X_samples=np.random.uniform(-20,20,10000)
y_samples=polynomail(X_samples)

# find minimum and maximum value of  generated dataset
X_min=np.min(X_samples)
X_max=np.max(X_samples)
y_min=np.min(y_samples)
y_max=np.max(y_samples)

#Normalize data use Min-Max scaling within range[-1,1]
X_normalized=2*(X_samples-X_min)/(X_max-X_min)-1
y_normalized=2*(y_samples-y_min)/(y_max-y_min)-1

#see normalized data
print(X_normalized)

#Split datasets for training, testing and validation using train_test_split for random split
#Initially keep 80% for training and keep rest 20% as temporary and then again divide the tempory datasets where 50% of temporary as testing and 50% of temporary as validation set
X_train,X_temp,y_train,y_temp=train_test_split(X_normalized,y_normalized,test_size=0.20,random_state=4)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.50,random_state=4)

#Here build the DNN
inputs=Input(shape=(1,),name="InputLayer")
X=Dense(32,activation='relu',name="HiddenLayer1")(inputs)
X=Dense(64,activation='relu',name="HiddenLayer2")(X)
X=Dense(128,activation='relu',name="HiddenLayer3")(X)
outputs=Dense(1,activation='linear',name="OutputLayer")(X)

#Load Model
model=Model(inputs=inputs,outputs=outputs)

#Show model architecture and total trainable parameter
model.summary()

#compile model for loss use mse and metrics use R2Score 
model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001),metrics=[metrics.R2Score(name='accuracy')])

#Use earlyStopping of callbacks function to reduce overfitting
early_stopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

#Train the model
history=model.fit(X_train,y_train,epochs=200,batch_size=32,validation_data=(X_val,y_val),callbacks=[early_stopping])

#evaluate model on test dataset
results=model.evaluate(X_test, y_test)
print(f"Raw evaluation results:{results}")
test_loss=results[0]
test_accuracy=results[1]

print(f"Test Loss:{test_loss}")
print(f"Test Accuracy:{test_accuracy}")

#Plot training error vs validation error
plt.plot(history.history['loss'],label='Training Error')
plt.plot(history.history['val_loss'],label='Validation Error')
plt.title('Training Error vs Validation Error (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#plot training accuracy vs validation accuracy
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Training Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()