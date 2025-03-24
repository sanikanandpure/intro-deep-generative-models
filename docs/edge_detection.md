---
title: Edge Detection
layout: default
nav_order: 6
---

# Edge Detection
Edges are crucial for visual perception, both for humans and machines. Humans primarily perceive images and objects in terms of edges. For example, if I gave you edges that correspond to an image of an animal (like below), you would very easily be able to identify that the animal in the picture is a cat.
![img1](edge_detection_img1.png)

This is possible because edges capture the most important structural information in an image. 

Edge detection--that is, extracting edges from a given image, is useful for a variety of applications. Identifying edges allows machines to extract meaningful features from images. It can also allow us to compress images into a smaller form by simply storing the extracted edges.

In deep learning models, particularly CNNs (which we'll talk about later), edges also serve as low-level features that help identify key aspects of an image. The earlier layers in a CNN often learn edges first, while later layers learn more complex features that build off of this ability to detect edges.

![img2](edge_detection_img2.png)

## What causes an edge?
Edges can be caused by a variety of visual reasons. Here are some of them:
- Changes in material properties: for example, there will be an edge between a rough material and a smooth material, if we laid them side by side
- Discontinuities in depth: recall that images are 2D representations of a 3D scene. In a 3D scene, depth is a separate dimension. In a 2D image, we can't really have depth. Thus, there will be an edge between an object and its background. 
- Variations in scene illumination/shadow: shadows and other differences in lighting cause edges!
- Discontinuities in surface orientation: once again, since an image is a 2D representation of a 3D scene, anytime an object wraps around itself, and the backside of the object is not visible, that discontinuity in the surface orientation (in the surface we are able to see) creates an edge. This might make more sense looking at the picture below. The column is wrapping around itself, creating a surface discontinuity.

![img3](edge_detection_img3.png)

At their core, edges occur when there is a large difference in pixel intensities of pixels on either side of the edge. We notice this in the image above; if you zoom in, you'll notice that edges mark a place of rapid change in intensity of pixels on either side.

![[edge_detection_img4.png]]


