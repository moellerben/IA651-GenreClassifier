# IA651-GenreClassifier
### Ben Moeller, Luke Buckler

For this project, we are trying to use machine learning techniques to classify genres of music.

## Data Source
Our data source for this project is the music library of WTSC, Clarkson's FM radio station. With around 15,000 songs in their library, many of which already labeled with genre metadata tags, this made for a perfect data source. In addition, the music library is mapped as a network drive for authorized users, making it super easy to access using Python.

## Data Preparation
In order to classify the audio data, we took the approach of converting audio to image, then using image classification techniques already well studied by the machine learning community. The audio to image conversion was done by generating a spectrogram from the song, which can be visualized by plotting frequency on the vertical axis, time on the horizontal axis, and intensity as brightness of each pixel. Since audio is perceived on a logarithmic scale, the spectrogram was scaled accordingly so a decent range of values is visible.

Since using every sample of every song proved to be way too much data, we opted to take just 5 1-second long clips from each song. We also created a second data file containing 2-second long clips to see how it impacts the accuracy of the model. Since the sample rate and number of frequency bins in the fast Fourier transform are consistent, the dimensionality of each clip is also consistent (for each clip duration).

The label for this process was taken directly from the "genre" metadata field within each file. Audio files that did not have any content in the "genre" tag were ignored. Many publishers and artists have different names for similar genres, so we created a mapping from each genre name appearing in the song tags to a consistent set of 10 genre names. The final dataset contains 8,669 songs with 5 clips each for a total of 43,345 clips. Each clip is stored as an image in a directory structure which indicates duration, train/test, and class.

## ML Techniques
Initially, our plan was to convert the images of the spectrographs into an array of intensities for each pixel, and then flatten that array into a row in a csv for each sample. The problem that we ran into for this was that in order to run the data through a CNN, we had to reshape the arrays in memory for every iteration of the CNN training and this produced many errors and was too taxing for us to run. We then pivoted to keeping the images as images and creating a folder structure storing all the images in folders based on music genre, and then setting up directories for where the training and testing data was. By doing this we were able to make the model training much less computationally difficult. For our model, we tested a number of different CNN architectures, as well as various parameters. We tried decreasing and increasing the batch size, increased the number of epochs but introduced the early stopping mechanic to save time where the difference in accuracy would be very small. As well, we added batch normalization between Conv2D steps to increase the speed of the model training. We also experimented with different optimizers but found adam to be the best performing option. In the end we went with the following model architecture:

Model: "sequential_6"

_________________________________________________________________

 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_18 (Conv2D)          (None, 127, 96, 16)       160       
                                                                 
 batch_normalization_12 (Bat  (None, 127, 96, 16)      64        
 chNormalization)                                                
                                                                 
 conv2d_19 (Conv2D)          (None, 125, 94, 16)       2320      
                                                                 
 max_pooling2d_12 (MaxPoolin  (None, 62, 47, 16)       0         
 g2D)                                                            
                                                                 
 conv2d_20 (Conv2D)          (None, 60, 45, 32)        4640      
                                                                 
 batch_normalization_13 (Bat  (None, 60, 45, 32)       128       
 chNormalization)                                                
                                                                 
 conv2d_21 (Conv2D)          (None, 58, 43, 32)        9248      
                                                                 
 max_pooling2d_13 (MaxPoolin  (None, 29, 21, 32)       0         
 g2D)                                                            
                                                                 
 flatten_5 (Flatten)         (None, 19488)             0         
                                                                 
 dropout_5 (Dropout)         (None, 19488)             0         
                                                                 
 dense_6 (Dense)             (None, 10)                194890    
                                                                 
=================================================================

Total params: 211,450
Trainable params: 211,354
Non-trainable params: 96
_________________________________________________________________

## Results
