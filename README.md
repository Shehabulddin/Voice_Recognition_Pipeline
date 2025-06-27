to run model.pynb set the image path to the path of the spectrograms, then run classifier and the cell under it, after that each of the models can be generated and compared
----------------------------------------------------------------------------------------------------------------------------------------
Interpretability Analysis of a Convolutional Neural Network

This Jupyter Notebook file explores the interpretability of a Convolutional Neural Network model trained on spectrogram images. It focuses on analyzing the network’s learned filters, feature maps, and similarity-based retrieval to better understand its decision-making process.

Features
Filter Visualization: Displays the learned filters from convolutional layers to examine how the network detects patterns.
Feature Map Analysis: Extracts and visualizes feature maps to observe activations at different layers.
Similarity Search: Uses cosine similarity on extracted embeddings to find the most similar spectrograms.
Model Architecture: A deep CNN with multiple convolutional layers, pooling layers, and fully connected layers trained on spectrogram data.

Files and Directories
model.h5 – The trained CNN model.
spectros_latest/ – Directory containing spectrogram images.
test_set.csv – CSV file with test data references.

Libraries needed:
numpy
pandas
matplotlib
PIL (Pillow)
scikit-learn
tensorflow or keras


--------------------------------------------------------------------------------------------------------------------------------------


Spectrogram Generation (spectgen.ipynb)

Audio Preprocessing
Silence Removal: Frames with low energy are removed based on a threshold.
Noise Reduction: Uses the first 0.5 seconds as a noise profile for suppression.
Segmentation: Audio is split into 3-second chunks.
Spectrogram Generation: Each segment is converted into a grayscale PNG spectrogram without axes.
Metadata Logging: Last segment duration is recorded in segmentlast_times.txt.

Folder Structure
Raw Audio: Stored in daps_audio/cleanraw/.
Saved Spectrograms: Saved in daps_images/cleanraw_images/sec3_noaxes/.

Unprocessed Spectrograms for EDA
Spectrograms are generated from raw audio without preprocessing steps.

Class 1 Augmentation (Time Stretching)
Files f1_, f7_, f8_, m3_, m6_, m8_ were time-stretched by a factor of 0.35.
Spectrograms generated from these augmented samples follow the same preprocessing steps.

Exploratory Data Analysis
Metadata: The segmentlast_times.txt file log experiments tracking time reduction after cleaning.
Comparison: Analyzed cleaned vs. cleaned and augmented class 1 spectrograms.

All created spectrograms (processed, unprocessed, and augmented) and the combined spectrograms in sec3_combined, which will be used in subsequent stages, are provided in daps_audio.zip.


--------------------------------------------------------------------------------------------------------------------------------------

Monte Carlo Dropout & Ensemble CNN for Uncertainty Estimation

Overview:
Monte Carlo Dropout (MC Dropout) – Uses dropout during inference to generate multiple predictions per sample.
Ensemble CNN – Trains multiple independent models and combines their predictions to estimate uncertainty.

How It Works:

Monte Carlo Dropout (MC Dropout)
A CNN model is trained with dropout layers active.
At inference time, dropout remains enabled, and the model runs multiple forward passes on the same input.
The mean prediction is taken as the final output, while the standard deviation of predictions represents uncertainty.

Ensemble CNN
Multiple CNN models (trained separately) are combined into an ensemble.
Each model makes a prediction, and the final output is the mean of all predictions.
The standard deviation of predictions represents uncertainty.

Files in This Repository:
Monte Carlo Dropout.ipynb – Jupyter Notebook with:
MC Dropout implementation.
Ensemble CNN implementation.
Training and evaluation of both methods.
Uncertainty visualization using histograms.

Exploratory Data Analysis (EDA)

The EDA - statistics.ipynb contains functions for analyzing key statistical properties of the dataset. This helps understand the distribution of pixel values and potential biases before training models.

Functions Included:
Mean Pixel Value Distribution – Computes the average intensity per image.
Pixel Variance Distribution – Measures how much pixel values vary within an image.
Entropy Calculation – Estimates the randomness or complexity of pixel distributions.
Skewness – Measures asymmetry in pixel value distribution.
Kurtosis – Identifies whether pixel distributions have heavy or light tails.
Dominant Frequency Analysis – Extracts the most significant frequency component in image data.

--------------------------------------------------------------------------------------------------------------------------------------

