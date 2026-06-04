### Procedure

The objective of this procedure is to implement a Generative Adversarial Network (GAN) using a Deep Convolutional GAN (DCGAN) architecture on the MNIST dataset. The experiment involves training a generator and discriminator in an adversarial manner to learn the data distribution and generate digit-like images. The procedure focuses on understanding adversarial training dynamics, monitoring generator and discriminator losses, and visually comparing real and generated images across epochs using a "Real vs Generated" panel to evaluate model performance.

---

#### 1. Environment Setup

Import the required Python libraries: PyTorch, Torchvision, NumPy, and Matplotlib.

---

#### 2. Dataset Preparation

- Load the MNIST handwritten digit dataset.
- Apply pre-processing steps, including resizing images to 16×16, converting them to tensors, and normalizing pixel values to the range [−1, 1].
- Separate the dataset into individual subsets for each digit (0–9).
- For each digit class, select a fixed number of samples and create separate DataLoaders to train the GAN independently on each digit.

---

#### 3. Hyper-parameter Initialization

- Define key hyperparameters: latent vector size, number of training epochs per digit, learning rate, and Adam optimizer momentum term.
- Set binary cross-entropy loss as the adversarial loss function.

---

#### 4. Model Architecture Definition

- Design the Generator using transposed convolutional layers, batch normalization, and ReLU activations to transform random noise into digit images.
- Design the Discriminator using convolutional layers with LeakyReLU activations to classify images as real or fake.
- Apply weight initialization to both networks to ensure stable training.

---

#### 5. Training Strategy (Digit-wise GAN Training)

For each digit from 0 to 9:

- Initialize a new Generator and Discriminator.
- Assign Adam optimizers to both networks.
- Cache a fixed batch of real images for visualization purposes.
- Train the GAN for a fixed number of epochs using alternating optimization:
  - Update the Discriminator by minimizing loss on real and fake samples.
  - Update the Generator by minimizing its loss to fool the Discriminator.
- Use label smoothing to improve training stability.

---

#### 6. Epoch-wise Monitoring and Visualization

- Compute average Generator and Discriminator losses for each epoch.
- Generate sample images using fixed noise vectors.
- Display a "Real vs Generated" image panel at regular intervals to visually evaluate learning progress.
- Assign a simple quality indicator based on smoothed generator loss values.

---

#### 7. Image Quality Evaluation

After each epoch, compute quantitative quality metrics:

- Pixel diversity
- Image sharpness
- Image contrast

Store loss values and quality metrics for later analysis.

---

#### 8. Result Visualization

- Load all trained generators and generate sample images for every digit.
- Display all generated digits together in a single grid for comparison.

---

#### 9. Training Metrics Analysis

- Plot Generator and Discriminator loss curves for all digits.
- Visualize image quality metrics across epochs to analyse training behaviour.

---

#### 10. Final Training Summary

- Display final loss values and quality metrics for each digit.
- Categorize training performance as Excellent, Good, or Needs Improvement.

---

#### 11. Interactive Sample Generation

- Implement an interactive function to generate images for any selected digit.
- Allow the user to specify the number of samples.
- Display generated images along with corresponding quality metrics.

---

#### 12. Result Analysis

- Generate samples for all digits to verify overall GAN performance.
- Observe diversity and realism in the generated handwritten digits.