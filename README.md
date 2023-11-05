# CML
The code repository of the article "Cascadic Multi-Receptive Learning for Multispectral Pansharpening".

Main contributions

1. A CML-resblock (see Sec. \ref{sec:CML}) is proposed to extract information from different scales in a step-by-step manner. Specifically, every pixel of the output is able to perceive multi-scale information through a cascade-like connection strategy, which is an efficient and effective multi-receptive learning process.
![comparisonforconv](https://github.com/wajuda/CML/assets/112617153/84d37822-6355-4978-91fb-3557dd2a4e4d)

2. Inspired by the traditional multiplicative injection model for pansharpening, we design the novel multiplication network structure (see Sec. \ref{sect:multi}) to learn the coefficients of the restoration mapping.
![network](https://github.com/wajuda/CML/assets/112617153/96c5066d-fd8a-474d-917d-0789e6ede797)
