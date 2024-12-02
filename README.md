## Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective
This paper introduces a new perspective by exploring adversarial vulnerability to frequency-selective perturbations in the spectral domain. A flexible attack framework is proposed to compare spatial and spectral domain attacks. Experiments show that Transformers rely heavily on phase and low-frequency information, making them more susceptible to frequency-based attacks than CNNs. 

<a href="https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Exploring_Adversarial_Robustness_of_Vision_Transformers_in_the_Spectral_Perspective_WACV_2024_paper.pdf" target="_blank">[paper]</a> 
<a href="https://www.youtube.com/watch?v=TP4MKRKGnp0" target="_blank">[video]</a> 

<img src="https://github.com/gihyunkim/exploring_adversarial_examples_in_spectral_perspective/blob/main/imgs/fourier_attack.png" width="700" height="300">


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
