# Location-Controlled-Brain-Tumour-Image-Synthesis
Course project for Medical Image Computing (CS736)

Generating location controlled abnormality in brain images using Controllable GANs,
extendable to other medical imaging applications like pneumonia, retinopathy, etc

## The Pipeline

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/pipeline.png)

### Rectifier

Normal Brain images are not readily available as it involves interaction with magnetic waves so healthy brain images are scarce. Given a MRI image with a tumour, the corresponding healthy brain image is even rarer.
To deal with above, we came up with a rectifier network which performs image in-painting on a brain image with the tumour part cropped out that generates the corresponding healthy image.

The key idea is to treat the tumour as a outlier in the image. That is, if we consider a bounding box at some location, then the event of this bounding box to have a tumour is an outlier. We exploit this property of Neural Networks to develop a rectifier. 

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/rectifier1.png)

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/rectifier2.png)

### Tumour Insertion

For this section, we will assume that we have tumour images, and the corresponding healthy image (given we have the rectifier). Given a healthy image and a custom bounding box, we need to insert a tumour at the specified location.

For the Adversarial Loss, we have trained both lsGAN as well as vanilla GAN. Both of the training processes give comparable results. We tried 3 variations in the loss function here:

- L1 loss
- L1 loss for the tumour part, MSE loss for the rest (1:1 ratio weight)
- L1 loss for the tumour part, MSE loss for the rest (1:100 ratio weight)

We have shown and compared the results for all 4 methods for the same healthy image, with different bounding boxes.

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/tumour1.png)

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/tumour2.png)

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/tumour3.png)

![](https://github.com/nirajmahajan/Location-Controlled-Brain-Tumour-Image-Synthesis/blob/master/images/tumour4.png)



