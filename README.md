# Defed
Defed: An Edge Feature Enhanced Image DenoisedNetworks Against Adversarial Attacks for SecureInternet-of-Things
### Abstract:
With the prosperous development of Internet of Things (IoT), IoT devices have been deployed in various applications, which generates large volume of image data to trace and record the users’ behaviors, resulting in better IoT services. To accurately analyze these huge data to further improve users’ experience on IoT services, deep neural networks (DNNs) are gaining more attention and have become increasingly popular. However, recent studies have shown that DNN models are vulnerable to adversarial attacks, which leads to the risk of applications in practice. Previous works are devoted to extract invariant features from the content circled by edges in images, while such features can not efficiently deal with the adversarial effect. In this work, we first study this problem from a new angle by exploring the edge feature information, which is intractable to be influenced by adversarial attacks demonstrated by our empirical analysis. Based on this, we propose a novel edge feature enhanced defense approach called Defed which incorporates edge feature information into denoised network to defend against various adversarial attacks in image area. For training phase, we only add benign images as the input and exert Gaussian noise to substitute the adversarial attacks to mitigate the dependency of models on specific adversarial attacks. For inference, we design a combination of multiple Defeds trained by different Gaussian noise levels and deploy confidence intervals to judge whether an image is adversarial or not. Experiments over real-world datasets on image classification demonstrate the efficacy and superiority compare to the state-of-the-art defense approaches.

**Published in: IEEE Internet of Things Journal**

paper: https://ieeexplore.ieee.org/document/9983796


## Code
This repository contains all the code needed to run the Defed. 

The implementation is based on **Pytorch**.

In particular:

