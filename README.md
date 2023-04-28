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


## Results