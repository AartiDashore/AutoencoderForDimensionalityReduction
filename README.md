# Autoencoder for Dimensionality Reduction

This project demonstrates how to build, train, and visualize an autoencoder for dimensionality reduction using `TensorFlow` and `Keras`. An autoencoder is a type of neural network used to learn efficient data representations (encoding) in an unsupervised manner. This project uses the MNIST dataset to show how the autoencoder compresses the data into a lower-dimensional representation and reconstructs it back.

## Features
- **Autoencoder Architecture**: Consists of an encoder and a decoder network.
- **Dimensionality Reduction**: Compresses 784-dimensional image vectors to a 32-dimensional representation.
- **Reconstruction**: Decodes the compressed representation back to its original size.
- **Visualization**: Compares the original and reconstructed images to evaluate the performance of the model.

## Concepts
- **Autoencoders**: Neural networks used for unsupervised learning of efficient representations.
- **Dimensionality Reduction**: Reducing the number of features in a dataset while retaining important information.
- **Feature Compression**: Encoding high-dimensional data into a lower-dimensional space.
  
## Prerequisites
Make sure to have the following installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the dependencies using pip:

```bash
pip install tensorflow keras numpy matplotlib
```

## Dataset
The project uses the MNIST dataset for training and testing the autoencoder. MNIST consists of 60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels.

## Model Architecture
- **Input Layer**: 784 neurons (flattened 28x28 images).
- **Encoder**: Reduces the dimensionality to 32 using a dense layer with ReLU activation.
- **Decoder**: Reconstructs the original input from the 32-dimensional encoded features using a dense layer with sigmoid activation.

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/autoencoder-dimensionality-reduction.git
   cd autoencoder-dimensionality-reduction
   ```

2. **Run the Autoencoder Script**:

   The script `autoencoder.py` builds and trains the model, and visualizes the results. You can run the script as follows:

   ```bash
   python autoencoder.py
   ```

3. **Training**:

   The autoencoder is trained on the MNIST dataset for 50 epochs with a batch size of 256. The training process will output the loss and validation loss at each epoch.

4. **Visualization**:

   After training, the script will plot a comparison of original and reconstructed images to show how well the autoencoder has compressed and reconstructed the data.

## Example Output

The output will display two rows of images:
- The top row shows the original images from the test dataset.
- The bottom row shows the reconstructed images from the autoencoder.

Example plot:

Epochs Generated:

![Epochs_1](https://github.com/AartiDashore/AutoencoderForDimensionalityReduction/blob/main/epochs_1.png)

![Epochs_2](https://github.com/AartiDashore/AutoencoderForDimensionalityReduction/blob/main/epochs_2.png)

Original vs Reconstructed Images

![Original vs Reconstructed Images](https://github.com/AartiDashore/AutoencoderForDimensionalityReduction/blob/main/Output_reduced_dimensionality.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
