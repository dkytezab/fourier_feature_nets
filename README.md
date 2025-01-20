# Fourier Feature Networks
Recent work in CV has explored the utility of replacing traditional, discrete representations of images with coordinate-based MLPs, fully-connected networks trained to learn a single image. They are compact and offer continuous representations of an image, making up- and down-scaling images easier. 

However, like other deep networks, coordinate-based MLPs suffer from 'spectral bias', an inability to learn high-frequency components of a given target function. Spectral bias, put forward as an explanation for generalization in high-dimensional settings, thus prevents these MLPs from learning fine-grain patterns in an image. To mitigate this, we apply a random feature embedding to our training coordinates, which can be interpreted as a composition of kernels – specifically, a composition of the neural tangent kernel and the radial basis function. We demonstrate how a) this composition enables tuning of the underlying MLP via choice of sampling procedure and b) imbues the MLP with shift-invariance for both 1-D toy functions and 2-D images. For a more comprehensive explanation, please consult ```exposition.pdf```

![alt text](https://github.com/dkytezab/fourier_feature_nets/blob/main/images/dog_comparison.png?raw=true)

We also demonstrate how these Fourier Feature Networks can be easily used to upscale images. As an example, we upscale a sample image of a dog to 9x its original pixel resolution.

<p align="center">
  <img width="300" height="400" src="https://github.com/dkytezab/fourier_feature_nets/blob/main/images/dog_upscaled.png">
</p>
