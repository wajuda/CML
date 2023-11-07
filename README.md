# Cascadic Multi-Receptive Learning for Multispectral Pansharpening

## How to use?
- Directly run: ``test.py`` for the single WV3 example
- Directly run: ``test_mulExm.py`` for the multiple WV3 examples

## Citation
```
@ARTICLE{10308614,
  author={Wang, Jun-Da and Deng, Liang-Jian and Zhao, Chen-Yu and Wu, Xiao and Chen, Hong-Ming and Vivone, Gemine},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Cascadic Multi-Receptive Learning for Multispectral Pansharpening}, 
  year={2023},
  doi={10.1109/TGRS.2023.3329881}}
```

## Motivations
### 1. CML-resblock
#### A CML-resblock (see Sec. \ref{sec:CML}) is proposed to extract information from different scales in a step-by-step manner. Specifically, every pixel of the output is able to perceive multi-scale information through a cascade-like connection strategy, which is an efficient and effective multi-receptive learning process.
![comparisonforconv](https://github.com/wajuda/CML/assets/112617153/84d37822-6355-4978-91fb-3557dd2a4e4d)

### 2. Multiplication network structure
#### Inspired by the traditional multiplicative injection model for pansharpening, we design the novel multiplication network structure (see Sec. \ref{sect:multi}) to learn the coefficients of the restoration mapping.
![network](https://github.com/wajuda/CML/assets/112617153/96c5066d-fd8a-474d-917d-0789e6ede797)

## Datasets

### 1. training datasets

[PanCollection](https://liangjiandeng.github.io/PanCollection.html)

### 2. testing datasets

[pan.baidu](https://pan.baidu.com/s/1UouPDZKYc8F_qBR4iPaD5g?pwd=d5pc)

## Results

### 1. Quantitative results

#### a. Single example

![HPBX2 7O){T{LQRQH2JG408](https://github.com/wajuda/CML/assets/112617153/f8d91d81-272c-4e16-a9f8-bab834db3b81)


#### b. Multiple examples

![DM QYX`8}0{(EXM`{%T27(H](https://github.com/wajuda/CML/assets/112617153/0b7523f8-f27b-4d72-a973-30f081bced62)


### 2. Visual results

#### a. Reduced resolution

 <img width="450" alt="d29542beb8be882172dcd43a74881d7" src="https://github.com/wajuda/CML/assets/112617153/c43a2086-a58e-4884-9b5a-3b381dcd7873" title="PAN">
<img width="450" alt="597307223ad7bed6a2f0528c32adc77" src="https://github.com/wajuda/CML/assets/112617153/bc1b6ca1-3186-4181-aac2-bdd3ed660a6d" title="Result"> 

#### b. Full resolution
   
<img width="450" alt="9d1709936d2e387bd49440115c82f22" src="https://github.com/wajuda/CML/assets/112617153/4e0e61e4-7e60-43b5-bfa2-67c620e0645f" title="PAN">  <img width="450" alt="2" src="https://github.com/wajuda/CML/assets/112617153/f719ac9b-de24-4f9f-9070-a80f0fbf5bde" title="Result">


