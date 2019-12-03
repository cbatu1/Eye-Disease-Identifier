# Eye-Disease-Identifier

A desktop application implemented in Python that uses a convolutional neural network to predict a variety of eye diseases.

## Requirements ##

* Python 3.5+
* Keras
* TensorFlow
* OpenCV
* Kivy

## Running the Project ##

1. Install all the required modules listed above and ensure everything is up to date.
2. Ensure the `DATA_DIRECTORY` variable in the `Eye Disease Identifier - CNN.py` file is set with the correct file path to the `Eye_Images` file on your system.
3. Run `Eye Disease Identifier - CNN.py` to train the CNN model if you have made adjustments, however the model is already saved as `64x0-CNN.model`.
4. Run `main.py` to start up the application. 

## Future Contributions ##

The current data sets are small and would require more data to further increase the accuracy of the model. Other adjustments could include methods to balance out the data such as oversampling of majority classes. The GUI is fairly basic and will require modifications to improve user experience. 
