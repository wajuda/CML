# CML
The code repository of the article "Cascadic Multi-Receptive Learning for Multispectral Pansharpening".

## Main contributions

1. A CML-resblock (see Sec. \ref{sec:CML}) is proposed to extract information from different scales in a step-by-step manner. Specifically, every pixel of the output is able to perceive multi-scale information through a cascade-like connection strategy, which is an efficient and effective multi-receptive learning process.
![comparisonforconv](https://github.com/wajuda/CML/assets/112617153/84d37822-6355-4978-91fb-3557dd2a4e4d)

2. Inspired by the traditional multiplicative injection model for pansharpening, we design the novel multiplication network structure (see Sec. \ref{sect:multi}) to learn the coefficients of the restoration mapping.
![network](https://github.com/wajuda/CML/assets/112617153/96c5066d-fd8a-474d-917d-0789e6ede797)

## Visual result

1. Reduced resolution

<img width="450" alt="1" src="https://github.com/wajuda/CML/assets/112617153/f80aae2f-7c82-42b6-a6fb-a2b01574f917" title="PAN"> | <img width="450" alt="2" src="https://github.com/wajuda/CML/assets/112617153/f719ac9b-de24-4f9f-9070-a80f0fbf5bde" title="Result">

2. Full resolution
   
<img width="450" alt="9d1709936d2e387bd49440115c82f22" src="https://github.com/wajuda/CML/assets/112617153/4e0e61e4-7e60-43b5-bfa2-67c620e0645f" title="PAN"> | <img width="450" alt="597307223ad7bed6a2f0528c32adc77" src="https://github.com/wajuda/CML/assets/112617153/bc1b6ca1-3186-4181-aac2-bdd3ed660a6d" title="Result">


