---
title: Image Representation
layout: default
nav_order: 2
---

# Image Representation
## What are images?
Images are 2D representations of a 3D scene. When we see an object with our eyes, we see it in a 3-dimensional space. To represent that scene as an image, we have to map that 3-dimensional scene onto a 2-dimensional plane. There are three major ways of representing images. 
### Matrix form
Images can be represented in the form of a matrix, where each pixel of the image corresponds to some value $(i, j)$ in the matrix. The values in the matrix represent intensity values, which can range from 0-255, where 0 corresponds to no intensity (black), and 255 corresponds to the maximum intensity (white). Oftentimes, for simplicity, these values are normalized (divided by 255) so that they fall between the inclusive range 0 (black) and 1 (white). 
#### Grayscale images
Thus, representing grayscale images is quite straightforward. Since the pixels are either black, white, or some shade of gray in between, we only require one matrix to represent the image, as all "colors" in the image can be represented with one set of values. In computer vision, the set of numbers required to represent the color of a single pixel are called channels. The number of channels in an image tells us how many numbers are used to specify each pixel's color. For grayscale images, since we only need one value to represent the color and intensity of each pixel, we consider them to have one channel.
#### Color images
What about color images? How can we represent them? Color images are represented using **three channels: Red, Green, and Blue (RGB)**. RGB representation of color images comes from the biological fact that cones in the human eye (receptors) perceive light roughly in terms of these three colors
This means that when we represent color images in matrix form, we actually require three matrices, each representing the intensities for R, G, and B. These three matrices are essentially stacked on top of each other to create a single matrix representing the image, where each value $(i, j)$ is a tuple consisting of RGB intensities: $[red, green, blue]$. 

Matrices are often the most common representation of images, because they are the most intuitive to look at and understand. They also make operations on images a lot easier, since matrix arithmetic operations (such as addition, subtraction, scaling etc.) all still apply here. We will talk about basic image transformations in the next set of notes.
### Function representation
Matrices can also be represented as continuous functions, where feeding an input $(i, j)$ to the function $f$ representing the image will output the corresponding intensity for that point. The key difference between a matrix and function representation of an image is that the matrix representation is discrete, whereas a function representation is continuous. This makes the function representation useful for certain mathematical operations, such as calculating gradients (this will come into play in the section on edge detection).

Once again, for grayscale images this representation is straightforward:
$f(x, y) = i$
For color images, which have three channels, the function representation is **vector-valued**, meaning that it outputs a vector, which in this case will have three components:
$f(x, y) = (f_R(x,y), f_G(x, y), f_B(x, y))$
### Digital images
Digital images are nothing very different from what we've been talking about; they are a discrete, 2D, digital representation of the function representation; they consist of pixels that are **sampled and quantized**. "Sampled" in this case means that we don't take all the values from that continuous function/3D scene; instead, we just sample a subset of values where the size of that subset (the number of values we sample) is called the **resolution** of that image. "Quantized" means that the intensity of each pixel falls within some fixed range (in this case, 0-255), and occurs in uniform steps. By uniform steps, we mean that we cannot have an "in-between" intensity such as, say, $121.255$ (just an example); our intensities occur at uniform, quantized steps of 1.

In the next set of notes, we will cover the basic image transformation operations: point, local, and global.






