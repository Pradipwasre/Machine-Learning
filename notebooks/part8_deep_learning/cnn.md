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

---

*More topics coming next (padding, stride, batch normalization, and more)*