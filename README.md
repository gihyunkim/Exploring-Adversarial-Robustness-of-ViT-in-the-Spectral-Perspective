## Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective
The Vision Transformer has emerged as a powerful tool for image classification tasks, surpassing the performance of convolutional neural networks (CNNs). Recently, many researchers have attempted to understand the robustness of Transformers against adversarial attacks. However, previous researches have focused solely on perturbations in the spatial domain. This paper proposes an additional perspective that explores the adversarial robustness of Transformers against frequency-selective perturbations in the spectral domain. To facilitate comparison between these two domains, an attack framework is formulated as a flexible tool for implementing attacks on images in the spatial and spectral domains. The experiments reveal that Transformers rely more on phase and low frequency information, which can render them more vulnerable to frequency-selective attacks than CNNs. This work offers new insights into the properties and adversarial robustness of Transformers.

<img src="https://github.com/gihyunkim/exploring_adversarial_examples_in_spectral_perspective/blob/main/imgs/fourier_attack.png" width="700" height="300">
<a href="https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Exploring_Adversarial_Robustness_of_Vision_Transformers_in_the_Spectral_Perspective_WACV_2024_paper.pdf" target="_blank">[paper]</a>.
<a href="https://www.youtube.com/watch?v=TP4MKRKGnp0" target="_blank">[video]</a> 

## NIPS 2017 Adversarial Competition Dataset
<a href=https://github.com/rwightman/pytorch-nips2017-adversarial>[Download]</a>

## Getting Started
```shell script
python --model_name resnet50 --attack phase --save
```
## Possible Models
|Model Name|ID|
|------|---|
|ResNet50|resnet50|
|ResNet152|resnet152|
|ViT-B pretrained on ImageNet-1k|vit-b-1k|
|ViT-B pretrained on ImageNet-21k|vit-b-21k|
|ViT-L|vit-l|
|Swin-B|swin-b|
|DeiT-S with distillation|deit-s|
|DeiT-S w/o distillation|deit-s-nodist|

## Citing
```shell script
@inproceedings{kim2024exploring,
  title={Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective},
  author={Kim, Gihyun and Kim, Juyeop and Lee, Jong-Seok},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3976--3985},
  year={2024}
}
```
