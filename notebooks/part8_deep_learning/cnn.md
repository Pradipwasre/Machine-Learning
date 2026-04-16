# Convolutional Neural Networks (CNN) - Complete Notes

---

## What is a CNN?

A Convolutional Neural Network (CNN) is a type of deep learning model designed specifically to work with images. It learns to detect patterns in images automatically, without us telling it what to look for.

Think of a CNN like your own eyes and brain working together. When you look at a cat, your brain does not process the whole image at once. It first notices edges, then shapes, then specific features like ears and whiskers, and finally decides "that is a cat." A CNN does exactly the same thing, step by step.

---

## The Full Flow of a CNN

Before going deep into each part, here is the complete picture of how a CNN works from start to finish:

```
Input Image
    |
    v
Convolution Layer  -->  Extracts features using filters/kernels
    |
    v
ReLU Layer         -->  Removes negative values, keeps useful signals
    |
    v
Pooling Layer      -->  Shrinks the feature map, keeps the important parts
    |
    v
Fully Connected    -->  Combines all features and makes a decision
    |
    v
Output             -->  cat = 0.99 | dog = 0.01
```

Each of these layers is explained in full detail below.

---

## The Convolution Operation

### What does "convolution" mean?

In mathematics, convolution describes the process of combining two functions to produce a third function.

In our case:

- The **first function** is the image (our input)
- The **second function** is a small matrix called a **filter** or **kernel**
- The **output** is called a **Feature Map**

The formula is simple:

```
Image  x  Kernel  =  Feature Map
```

### Understanding with a Matrix Example

Let us say we have a small binary image (black and white, using only 0 and 1). A binary image uses 0 for black pixels and 1 for white pixels.

**Step 1 - The Input Image (5x5)**

This is our original 5x5 image:

```
Input Image (5x5)
+---+---+---+---+---+
| 1 | 0 | 1 | 0 | 1 |
+---+---+---+---+---+
| 0 | 1 | 1 | 1 | 0 |
+---+---+---+---+---+
| 1 | 1 | 0 | 1 | 1 |
+---+---+---+---+---+
| 0 | 1 | 1 | 0 | 0 |
+---+---+---+---+---+
| 1 | 0 | 1 | 1 | 0 |
+---+---+---+---+---+
```

**Step 2 - The Filter / Kernel (3x3)**

This is our filter (also called kernel). It is a small matrix we slide over the image:

```
Filter / Kernel (3x3)
+---+---+---+
| 1 | 0 | 1 |
+---+---+---+
| 0 | 1 | 0 |
+---+---+---+
| 1 | 0 | 1 |
+---+---+---+
```

**Step 3 - How the Multiplication Happens**

We place the filter on top of the top-left 3x3 region of the image. We multiply each value in the image with the matching value in the filter, then add all the results together. This gives us one number in the Feature Map.

```
Image region:           Filter:             Multiply and Sum:
+---+---+---+           +---+---+---+
| 1 | 0 | 1 |    x      | 1 | 0 | 1 |      (1x1)+(0x0)+(1x1)
+---+---+---+           +---+---+---+    +  (0x0)+(1x1)+(1x0)
| 0 | 1 | 1 |           | 0 | 1 | 0 |    +  (1x1)+(1x0)+(0x1)
+---+---+---+           +---+---+---+
| 1 | 1 | 0 |           | 1 | 0 | 1 |    =  1+0+1+0+1+0+1+0+0 = 4
+---+---+---+           +---+---+---+
```

We slide this filter one step at a time across the entire image, computing one value for each position.

**Step 4 - The Output Feature Map (3x3)**

After sliding the filter across all positions, we get this Feature Map:

```
Feature Map (3x3)
+---+---+---+
| 4 | 3 | 4 |
+---+---+---+
| 2 | 4 | 3 |
+---+---+---+
| 3 | 3 | 4 |
+---+---+---+
```

### Calculating Feature Map Size

There is a simple formula to know how big the Feature Map will be:

```
Feature Map Size  =  n - f + 1  =  m

Where:
  n  =  size of the input image
  f  =  size of the filter

Example:
  5 - 3 + 1  =  3
  (5x5 image with 3x3 filter gives a 3x3 Feature Map)
```

---

## Image Features

### What is a Feature Map, really?

Feature Maps are not just output numbers. They are **feature detectors**. Each Feature Map tells us "where in the image does this particular pattern appear?"

### Why do we do this?

Because filters are very good at detecting specific things in images:

- A filter with vertical lines detects vertical edges
- A filter with diagonal patterns detects diagonal edges
- More complex filters detect curves, corners, or textures

**Simple Example: Edge Detection**

Imagine you have a photo of a face. An edge-detecting filter will produce a Feature Map that is bright (high values) exactly where the face has clear outlines (around the nose, eyes, lips) and dark (low values) everywhere else.

This is useful because once we know where the edges are, we can build up to understanding shapes, and from shapes we can understand objects. A CNN learns these filters automatically during training.

**Benefits:**

- We do not need to manually tell the CNN what to look for
- The same filter works no matter where the feature appears in the image
- Multiple filters can detect multiple features at the same time

---

## Feature Detectors

### How Classifiers Use Feature Detectors

A classifier is the final part of the CNN that makes a decision about what the image contains. It uses the Feature Maps to decide the answer.

Here is how the full process works:

**Best Example - Classifying a Cat vs Dog:**

Imagine you are training a CNN on thousands of cat and dog photos.

1. The first filters learn simple things: ears have curved edges, snouts have certain shapes, fur has specific textures
2. As we go deeper, filters combine: "pointed ear on top of round head" = likely cat
3. All these Feature Maps are passed to the classifier
4. The classifier sees all the evidence and gives a probability score

```
Result:
  cat  =  0.99   (99% sure this is a cat)
  dog  =  0.01   (1% chance it is a dog)
```

The key steps are:

- We perform the convolution operation with the input image and our filters
- The filters (Conv filters) create Feature Maps
- These Feature Maps are passed through Hidden Layers
- The Hidden Layers combine all evidence
- The Output Layer gives us the final prediction

---

## The Layers of a CNN

A CNN is built from several layers, each with a specific job. Together they work like an assembly line.

### 1. Convolution Layer

This is where filters scan the image and create Feature Maps. Each filter detects one specific pattern. A typical CNN uses many filters at once (for example, 32 or 64 filters), so it detects 32 or 64 different features at the same time.

### 2. ReLU Layer

ReLU stands for Rectified Linear Unit. After convolution, some values in the Feature Map will be negative. ReLU removes all negative numbers and replaces them with zero.

Why? Because negative values do not help the CNN learn. A feature either exists or it does not.

```
Rule:  If value > 0, keep it.
       If value <= 0, make it 0.

Example:
  Before ReLU:  [-2, 4, 0, -1, 7, -3]
  After ReLU:   [ 0, 4, 0,  0, 7,  0]
```

### 3. Pooling Layer

Pooling shrinks the Feature Map to make it smaller. The most common type is Max Pooling, where we take the highest value from each small region.

Why shrink? Because we want to keep only the most important information and reduce the amount of data the next layer has to process.

```
Before Pooling (4x4):         After Max Pooling (2x2):
+---+---+---+---+             +---+---+
| 1 | 3 | 2 | 4 |             | 3 | 4 |
+---+---+---+---+     -->     +---+---+
| 5 | 2 | 8 | 1 |             | 8 | 9 |
+---+---+---+---+             +---+---+
| 3 | 1 | 9 | 2 |
+---+---+---+---+
| 4 | 2 | 1 | 3 |
+---+---+---+---+
```

(We took the max from each 2x2 block)

### 4. Fully Connected Layer (Dense Layer)

After convolution, ReLU, and pooling are done, the Feature Maps are "flattened" into a single long list of numbers. This list is then passed to the Fully Connected Layer.

A Fully Connected Layer is the same as a regular neural network. Every number is connected to every node in the next layer. This is where the final reasoning happens.

### 5. Output Layer

The final layer gives us the predictions. It outputs a probability for each possible class.

```
Example output for a 3-class problem:
  cat  =  0.90
  dog  =  0.08
  bird =  0.02
```

The class with the highest probability is the final answer.

---

## Convolution on Colour Images (RGB)

Everything so far assumed a grayscale (black and white) image. Real images have colour.

### How Colour Images Work

A colour image has 3 channels: Red, Green, and Blue (RGB). Instead of one 2D grid of numbers, we have three stacked 2D grids.

```
Colour Image = 3 separate 2D grids stacked on top of each other

  Red Channel:    Green Channel:   Blue Channel:
  +---+---+---+   +---+---+---+   +---+---+---+
  |255| 0 | 0 |   | 0 |128| 0 |   | 0 | 0 |255|
  |...|...|...|   |...|...|...|   |...|...|...|
  +---+---+---+   +---+---+---+   +---+---+---+
```

### The Filter Also Has 3 Channels

When we apply a filter to a colour image, the filter also has 3 layers (one for each colour channel):

```
Input Image (3 channels)  x  Filter (3 channels)  =  Feature Map (1 channel)

  Each channel of the image is multiplied with the matching
  channel of the filter. All results are added together
  to produce one single Feature Map.
```

### Why Does This Matter?

Because now our filters can detect features that depend on colour.

For example:

- A filter can detect red objects specifically (like a stop sign)
- A filter can detect green regions (like grass or trees)
- A filter can detect blue areas (like sky or water)

This means the CNN can learn features that are specific to a colour, which is much more powerful than only detecting edges and shapes.

### The Process in Steps

1. We start with the input image (height x width x 3 channels)
2. We apply a filter (3 x 3 x 3 - three layers of 3x3 grids)
3. Each layer of the filter scans the matching colour channel
4. All three results are summed to produce one value in the Feature Map
5. The final Feature Map is a 2D grid (just like with grayscale)
6. We use many filters, so we produce many Feature Maps (one per filter)

```
Input Image:           Filter:              Output:
   W x H x 3      x    3 x 3 x 3      =    Feature Map
   (colour)            (3-channel)          (2D, per filter)
```

---

## Summary Table

| Layer | What it Does | Input | Output |
|---|---|---|---|
| Convolution | Scans image with filters | Image | Feature Maps |
| ReLU | Removes negative values | Feature Maps | Cleaned Feature Maps |
| Pooling | Shrinks the maps | Feature Maps | Smaller Feature Maps |
| Fully Connected | Combines all evidence | Flattened vector | Class scores |
| Output | Final prediction | Class scores | Probabilities |

---

## Quick Recap

- A CNN reads images by sliding small filters across them
- Each filter detects one type of pattern and creates a Feature Map
- Feature Maps show where in the image a pattern was found
- Multiple layers build from simple patterns (edges) to complex ones (faces, objects)
- The final layers combine all evidence and output a class prediction with a confidence score

------

======================================

# Convolutional Neural Networks (CNN) - Complete Notes

---

## The Full Flow of a CNN

This is the complete path an image takes inside a CNN, from raw pixels to a final prediction.

```
Input Image
    |
    v
Feature Detectors (Filters / Kernels)
    |
    v
Convolution Layer  -->  Slides filters over the image, creates Feature Maps
    |
    v
Activation Function (ReLU)  -->  Removes negative values, keeps useful signals
    |
    v
Pooling Layer  -->  Shrinks Feature Maps, keeps only the strongest signals
    |
    v
Fully Connected Layer  -->  Combines all evidence from all Feature Maps
    |
    v
Softmax Output  -->  cat = 0.92 | dog = 0.06 | bird = 0.02
```

Each stage is explained in detail below, in the same order as the flow above.

---

## 1. Feature Detectors

A feature detector is a small matrix (also called a filter or kernel) that slides over the image. Its job is to find one specific pattern anywhere in the image.

**What counts as a feature?**

- A vertical edge
- A horizontal edge
- A diagonal line
- A curve or corner
- A texture (like fur or grass)

**How it works:**

The filter slides across the image one position at a time. At each position, it multiplies its values with the image values underneath it and adds everything up. The result is a single number. After sliding across the whole image, all these numbers form a Feature Map.

A Feature Map answers the question: "Where in this image does this specific pattern appear?"

**Key points:**

- One filter = one Feature Map
- A CNN uses many filters at once (for example, 32 or 64)
- So one convolution layer produces 32 or 64 Feature Maps, each detecting something different
- The CNN learns these filters automatically during training. We do not hand-design them.

---

## 2. 3D Convolutions and Color Images

A grayscale image is a 2D grid of numbers. A color image has three separate 2D grids stacked on top of each other, one for Red, one for Green, and one for Blue. This is called a 3-channel or 3D input.

**How the filter changes for color images:**

A filter must also have 3 channels to match the input. So instead of a 3x3 filter, we use a 3x3x3 filter (three layers of 3x3).

```
Input Image (H x W x 3)   x   Filter (3 x 3 x 3)   =   Feature Map (H x W x 1)
```

At each position, the filter scans all three color channels at once. Each channel of the filter multiplies with the matching channel of the image. All results from all three channels are added together to produce one single number in the Feature Map.

**Why this matters:**

- The CNN can now learn features that depend on color, not just shape
- For example, one filter can learn to detect red stop signs
- Another filter can learn to detect green grass
- Another can detect blue sky

**Output depth:**

Even though the input has 3 channels, one filter produces one 2D Feature Map. If we use 64 filters, we get 64 Feature Maps. The output has depth 64.

```
Input: H x W x 3
After 64 filters: H x W x 64
```

---

## 3. Kernel Size and Depth

**Kernel size** refers to the height and width of the filter matrix.

Common sizes:

| Kernel Size | What it is good for |
|-------------|---------------------|
| 1 x 1 | Adjusting depth, combining channels |
| 3 x 3 | Most common, captures local patterns |
| 5 x 5 | Slightly larger neighborhood |
| 7 x 7 | Used in the first layer to capture broad patterns |

Smaller kernels (3x3) are preferred today because stacking multiple 3x3 layers gives the same field of view as one large kernel, but with fewer parameters and more learning.

**Kernel depth** always matches the depth of the input.

- Input with 3 channels → filter has depth 3
- Input with 64 channels (from previous layer) → filter has depth 64

**Output size formula:**

```
Output Size = (n - f + 1)

Where:
  n = size of input
  f = size of filter (kernel)

Example: 28x28 input, 3x3 filter
  Output = 28 - 3 + 1 = 26
  Output size = 26 x 26
```

---

## 4. Padding

When a filter slides over an image, the output Feature Map is smaller than the input. Also, the pixels at the edges of the image are visited fewer times than pixels in the center, so edge information is lost.

**Padding** solves both problems by adding a border of zeros around the image before convolution.

**Two types:**

**Valid Padding (no padding):**
- No zeros are added
- Output is smaller than input
- Output size = n - f + 1

**Same Padding:**
- Zeros are added around the border
- Output size stays the same as the input size
- Amount of padding needed = (f - 1) / 2

```
Example:
  Input: 5 x 5
  Filter: 3 x 3
  Padding: 1 (one row/column of zeros on each side)
  Output: 5 x 5 (same as input)
```

**Why use padding:**

- Keeps spatial dimensions from shrinking after every layer
- Allows deeper networks without the image becoming too small
- Preserves information at the edges of the image

---

## 5. Stride

**Stride** is how many steps the filter moves each time it slides across the image.

- Stride = 1: Filter moves one pixel at a time (standard, more overlap)
- Stride = 2: Filter moves two pixels at a time (faster, less overlap, smaller output)

**Output size formula with stride:**

```
Output Size = floor((n - f + 2p) / s) + 1

Where:
  n = input size
  f = filter size
  p = padding
  s = stride

Example: n=7, f=3, p=0, s=2
  Output = floor((7 - 3 + 0) / 2) + 1 = floor(4/2) + 1 = 3
```

**Why stride matters:**

- Stride 1 keeps more spatial detail
- Stride 2 or more reduces the output size, similar to pooling, but it does it during the convolution step
- Larger stride means fewer computations, smaller Feature Maps

---

## 6. Activation Functions

After convolution, each value in the Feature Map is passed through an activation function. This introduces non-linearity into the network, which means the CNN can learn complex patterns, not just straight-line relationships.

**Why non-linearity is needed:**

Without activation functions, stacking many layers is mathematically the same as having just one layer. The network cannot learn complex things.

**ReLU (Rectified Linear Unit) - Most Common:**

```
Rule:
  If value > 0, keep it
  If value <= 0, set it to 0

Example:
  Input:  [-3,  5, -1,  8,  0, -2]
  Output: [ 0,  5,  0,  8,  0,  0]
```

ReLU is fast, simple, and works very well in practice. It removes all negative values, which means "this feature was not found here."

**Leaky ReLU:**

Instead of setting negative values to zero, it multiplies them by a small number (like 0.01). This prevents neurons from becoming permanently inactive.

```
Rule:
  If value > 0, keep it
  If value <= 0, multiply by 0.01
```

**Sigmoid:**

Maps any value to a range between 0 and 1. Mostly used in the output layer for binary classification (yes or no problems).

**Tanh:**

Maps values to a range between -1 and 1. Sometimes used in hidden layers.

**In CNNs, ReLU is used after every convolution layer. Sigmoid or Softmax is used only in the final output layer.**

---

## 7. Pooling

Pooling reduces the size of the Feature Map. It keeps the most important information and discards the rest.

**Why pooling is needed:**

- Reduces the number of parameters the network has to learn
- Speeds up training
- Makes the network less sensitive to the exact position of a feature (spatial invariance)

**Max Pooling (most common):**

Divides the Feature Map into small regions (for example 2x2) and keeps only the maximum value from each region.

```
Before Max Pooling (4 x 4):        After Max Pooling (2 x 2), stride=2:
+---+---+---+---+                   +---+---+
| 1 | 3 | 2 | 4 |                   | 5 | 8 |
+---+---+---+---+         -->        +---+---+
| 5 | 2 | 8 | 1 |                   | 4 | 9 |
+---+---+---+---+                   +---+---+
| 3 | 1 | 9 | 2 |
+---+---+---+---+
| 4 | 2 | 1 | 3 |
+---+---+---+---+

Top-left 2x2:  max(1,3,5,2) = 5
Top-right 2x2: max(2,4,8,1) = 8
Bottom-left 2x2: max(3,1,4,2) = 4
Bottom-right 2x2: max(9,2,1,3) = 9
```

**Average Pooling:**

Takes the average of values in each region instead of the maximum. Used less often but sometimes applied in the final layers.

**Global Average Pooling:**

Takes the average of the entire Feature Map and produces a single number per map. Often used right before the Fully Connected Layer in modern networks.

**Common pooling settings:**
- Pool size: 2 x 2
- Stride: 2
- This halves the width and height of the Feature Map

---

## 8. Fully Connected Layers

After all the convolution, activation, and pooling steps are done, the Feature Maps are flattened into a single long list of numbers (a 1D vector). This vector is then passed to one or more Fully Connected Layers.

**What "fully connected" means:**

Every number in the input vector is connected to every neuron in the next layer. This is the same structure as a standard neural network.

**What it does:**

- Combines all the evidence gathered from the Feature Maps
- Learns which combinations of features correspond to which class
- Acts as the final reasoning stage

**Example:**

```
Feature Maps (8 x 8 x 64) --> Flatten --> 4096 values --> FC Layer --> 256 neurons --> Output Layer
```

**Dropout:**

During training, Dropout randomly switches off some neurons in the Fully Connected Layer. This prevents the network from memorizing the training data (overfitting) and forces it to learn more general patterns.

---

## 9. Softmax

Softmax is the activation function used in the final output layer when the problem has more than two classes.

**What it does:**

It takes the raw output scores (called logits) and converts them into probabilities. The probabilities always add up to 1.

**Formula:**

```
Softmax(z_i) = e^(z_i) / sum of e^(z_j) for all j
```

**Example:**

```
Raw scores (logits):   [2.0,  1.0,  0.1]

After Softmax:
  cat  =  0.70   (70%)
  dog  =  0.26   (26%)
  bird =  0.04   (4%)

Total = 1.00
```

The class with the highest probability is the final prediction.

**Softmax vs Sigmoid:**

| | Sigmoid | Softmax |
|--|---------|---------|
| Output | One value between 0 and 1 | One probability per class |
| Use case | Binary classification (2 classes) | Multi-class classification (3+ classes) |
| Outputs sum to 1? | No | Yes |

---

## 10. Putting Together Your Convolutional Neural Network

A complete CNN is built by stacking the layers in a repeating pattern, followed by the classifier at the end.

**Typical CNN structure:**

```
Input Image
    |
[Conv --> ReLU --> Pooling]   Block 1 (learns simple features: edges, lines)
    |
[Conv --> ReLU --> Pooling]   Block 2 (learns medium features: shapes, curves)
    |
[Conv --> ReLU --> Pooling]   Block 3 (learns complex features: eyes, wheels, etc.)
    |
Flatten
    |
Fully Connected + ReLU
    |
Dropout (optional, for regularization)
    |
Fully Connected
    |
Softmax Output
```

**Design decisions to make:**

- How many Conv blocks to use (more blocks = deeper network = learns more complex features)
- How many filters per Conv layer (32, 64, 128 are common starting points)
- Kernel size (3x3 is the standard today)
- Whether to use padding (Same padding keeps dimensions constant)
- Pool size and stride (2x2 with stride 2 is the standard)
- How many neurons in the Fully Connected Layer

**Example architecture for image classification (10 classes):**

```
Input: 32 x 32 x 3

Conv (32 filters, 3x3, Same padding)   -->  32 x 32 x 32
ReLU
Max Pool (2x2, stride 2)               -->  16 x 16 x 32

Conv (64 filters, 3x3, Same padding)   -->  16 x 16 x 64
ReLU
Max Pool (2x2, stride 2)               -->   8 x  8 x 64

Flatten                                -->  4096
Fully Connected (128 neurons)          -->   128
ReLU
Fully Connected (10 neurons)           -->    10
Softmax                                -->    10 probabilities
```

---

## 11. Parameter Counts in CNNs

A parameter is any number the CNN learns during training. Knowing how many parameters exist helps us understand the size and complexity of the model.

**Parameters in a Conv Layer:**

```
Parameters = f x f x d_in x d_out + d_out

Where:
  f      = filter size (e.g., 3)
  d_in   = depth of input (number of channels coming in)
  d_out  = number of filters (channels going out)
  + d_out = one bias per filter

Example:
  Conv layer: 3x3 filter, input depth 3, 32 filters
  Parameters = 3 x 3 x 3 x 32 + 32 = 864 + 32 = 896
```

**Parameters in a Fully Connected Layer:**

```
Parameters = inputs x outputs + outputs (bias)

Example:
  4096 inputs, 128 outputs
  Parameters = 4096 x 128 + 128 = 524,288 + 128 = 524,416
```

**Key observation:**

Conv layers have very few parameters compared to Fully Connected Layers. The table below shows why CNNs are efficient:

```
Layer                   Output Shape     Parameters
-----------             ------------     ----------
Conv (32, 3x3)          32 x 32 x 32         896
Max Pool                16 x 16 x 32           0
Conv (64, 3x3)          16 x 16 x 64      18,496
Max Pool                 8 x  8 x 64           0
Flatten                       4096           0
Fully Connected (128)          128     524,416
Output (10)                     10       1,290
```

Most parameters are in the Fully Connected Layers, not the Conv Layers. This is why modern CNNs often replace large Fully Connected Layers with Global Average Pooling.

---

## 12. Why CNNs Work So Well on Images

Standard neural networks treat every pixel as a separate, independent input. This ignores the spatial structure of the image and leads to enormous numbers of parameters.

CNNs solve this with three key ideas:

**1. Local Connectivity:**
Each filter only looks at a small region of the image at a time (for example, 3x3 pixels). This matches how real features in images are local. An ear is made of nearby pixels, not pixels scattered across the whole image.

**2. Weight Sharing:**
The same filter is applied to every position in the image. This means the CNN uses the same learned weights to detect, for example, a vertical edge whether it appears on the left, right, top, or bottom of the image. This dramatically reduces the number of parameters.

**3. Hierarchical Feature Learning:**
Early layers learn simple patterns (edges, colors). Middle layers combine these into shapes (curves, circles). Deeper layers combine shapes into objects (eyes, wheels, faces). This matches how human vision works.

**Summary of advantages:**

| Property | What it means |
|----------|---------------|
| Local Connectivity | Only nearby pixels are connected to each filter |
| Weight Sharing | One filter used across the whole image, fewer parameters |
| Translation Invariance | A feature is detected wherever it appears in the image |
| Hierarchical Learning | Simple patterns build into complex patterns layer by layer |

---

## 13. Training a CNN

Training is the process of adjusting all the filter weights and FC layer weights so the CNN makes correct predictions.

**The training loop (one cycle = one epoch):**

```
Step 1: Forward Pass
  - Feed a batch of images through the CNN
  - Get predictions (probabilities from Softmax)

Step 2: Compute Loss
  - Compare predictions to the correct labels
  - Calculate a number that measures how wrong the predictions are (the loss)

Step 3: Backward Pass (Backpropagation)
  - Calculate how much each weight contributed to the error
  - Compute gradients for every weight in the network

Step 4: Update Weights (Gradient Descent)
  - Adjust every weight slightly in the direction that reduces the loss
  - Repeat from Step 1
```

**Key terms:**

- **Epoch**: One full pass through the entire training dataset
- **Batch**: A small group of images processed together (e.g., 32 or 64 images at a time)
- **Iteration**: One forward + backward pass on one batch

---

## 14. Loss Functions

A loss function measures how wrong the CNN's predictions are. The goal of training is to minimize this number.

**Cross-Entropy Loss (most common for classification):**

Used when the output is probabilities from Softmax.

```
Loss = - sum of [ y_true x log(y_predicted) ]

Where:
  y_true     = 1 for the correct class, 0 for all others
  y_predicted = the probability the CNN gave to each class
```

**Intuition:**

- If the CNN is very confident and correct: loss is close to 0
- If the CNN is very confident but wrong: loss is very large

**Example:**

```
Correct label: cat
Predicted:  cat=0.90, dog=0.08, bird=0.02

Loss = -(1 x log(0.90) + 0 x log(0.08) + 0 x log(0.02))
     = -(log(0.90))
     = 0.105   (low loss, good prediction)

If predicted:  cat=0.05, dog=0.90, bird=0.05
Loss = -(log(0.05)) = 2.99   (high loss, bad prediction)
```

**Other loss functions:**

| Loss Function | When to use |
|---------------|-------------|
| Binary Cross-Entropy | Two classes (cat vs not-cat) |
| Categorical Cross-Entropy | Multiple classes (cat, dog, bird) |
| Mean Squared Error | Regression problems (predicting a number) |

---

## 15. Backpropagation

Backpropagation is the algorithm that calculates how much each weight in the network contributed to the final error. It does this by working backwards through the network, from the output layer to the input layer.

**The chain rule:**

Backpropagation uses the chain rule from calculus. The gradient of the loss with respect to an early layer's weights equals the gradient at the output multiplied by the gradient at each layer in between.

**Step by step:**

```
1. Forward pass: compute the output and calculate the loss

2. Compute gradient at the output layer
   How much does a small change in the output change the loss?

3. Pass gradient backwards through the Fully Connected Layers
   How much did each FC weight contribute to the output?

4. Pass gradient backwards through the Pooling Layer
   (Pooling passes the gradient only to the position of the max value)

5. Pass gradient backwards through the ReLU Layer
   (Gradient passes through where ReLU was active, blocked where ReLU was zero)

6. Pass gradient backwards through the Conv Layer
   How much did each filter weight contribute to the Feature Map?

7. Now we have a gradient for every single weight in the entire network
```

**What backpropagation gives us:**

A gradient for every parameter. The gradient tells us:
- The direction to change the weight (increase or decrease)
- How sensitive the loss is to that weight

---

## 16. Gradient Descent

Gradient Descent uses the gradients from backpropagation to update all the weights in the network.

**The update rule:**

```
new weight = old weight - (learning rate x gradient)

Or in symbols:
  w = w - lr x dL/dw

Where:
  w     = current weight
  lr    = learning rate (a small number, e.g., 0.001)
  dL/dw = gradient of the loss with respect to this weight
```

**Learning rate:**

The learning rate controls how big each update step is.

- Too large: weights change too fast, the training overshoots and does not converge
- Too small: training is very slow
- Typical values: 0.1, 0.01, 0.001, 0.0001

**Types of Gradient Descent:**

| Type | How it works |
|------|--------------|
| Batch Gradient Descent | Uses the entire dataset to compute one update (slow, accurate) |
| Stochastic Gradient Descent (SGD) | Uses one sample at a time (fast, noisy) |
| Mini-Batch Gradient Descent | Uses a small batch (e.g., 32 samples) - most common in practice |

**Optimizers (improved versions of Gradient Descent):**

Modern CNNs rarely use plain Gradient Descent. These optimizers adapt the learning rate automatically:

| Optimizer | Description |
|-----------|-------------|
| SGD with Momentum | Remembers previous update directions to smooth out the path |
| Adam | Adaptive learning rate for each weight, combines momentum and scaling |
| RMSProp | Adapts learning rate based on recent gradient magnitudes |

**Adam** is the most widely used optimizer for CNNs today.

**When does training stop?**

Training runs for a fixed number of epochs, or until:
- The validation loss stops improving
- Early stopping is triggered (automatically stops when performance plateaus)

---

## Quick Reference Summary

```
Component          Job
----------         ---
Filter / Kernel    Detects one specific pattern in the image
Feature Map        Shows where that pattern was found
Padding            Keeps output the same size as input
Stride             Controls how far the filter moves each step
ReLU               Removes negative values, adds non-linearity
Pooling            Shrinks the Feature Map, keeps strongest signals
Fully Connected    Combines all evidence, reasons about the class
Softmax            Converts scores into probabilities (sum = 1)
Loss Function      Measures how wrong the predictions are
Backpropagation    Calculates gradient of loss for every weight
Gradient Descent   Updates weights to reduce the loss
```

*More topics coming next (padding, stride, batch normalization, and more)*
