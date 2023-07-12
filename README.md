## CIF-PyTorch

:rocket: **Attention! Please refer to https://github.com/MingLunHan/CIF-HieraDist for the our latest and complete implementation of the CIF-based speech recognition model！**

A PyTorch implementation of Continuous Integrate-and-Fire (CIF) module for end-to-end (E2E) automatic speech recognition (ASR), which is originally proposed in **Cif: Continuous integrate-and-fire for end-to-end speech recognition** https://ieeexplore.ieee.org/document/9054250.

If you have any questions, please consult hanminglun1996@foxmail.com .

### 1. A Feasible Configuration for CIF Module
```
encoder_embed_dim: 256 # should be the innermost dimension of inputs
produce_weight_type: "conv"
cif_threshold: 0.99
conv_cif_layer_num: 1
conv_cif_width: 3 or 5
conv_cif_output_channels_num: 256
conv_cif_dropout: 0.1
dense_cif_units_num: 256
apply_scaling: True
apply_tail_handling: True
tail_handling_firing_threshold: 0.5
add_cif_ctxt_layers: False
```

### 2. **Tips**

1. For speech recognition, we usually down-sample the input frame sequence to 1/8 of the its length at the encoder side to ensure efficient training of the CIF module. For other tasks, it should also be ensured that the length difference between input and output of the CIF is kept within reasonable range.
2. During training, when the scaled sum of the weights differs from the length of the reference transcription, you can truncate the reference and the model output to the same length.
3. The scaling strategy during training stage may cause gradient exploding, because the calculation of normalize scalar needs division operation. You could add a small value (1e-8) to the denominator to avoid this problem.

### 3. **Other CIF Resources and Materials**

#### a. Papers:

**ASR**:
  - CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition https://ieeexplore.ieee.org/document/9054250
  - A Comparison of Label-Synchronous and Frame-Synchronous End-to-End Models for Speech Recognition https://arxiv.org/abs/2005.10113
  - Knowledge Transfer from Pre-trained Language Models to CIF-based Speech Recognizers via Hierarchical Distillation https://arxiv.org/pdf/2301.13003

**ASR Contextualization & Customization & Personalization**:
  - CIF-based Collaborative Decoding for End-to-End Contextual Speech Recognition https://ieeexplore.ieee.org/document/9415054
  - Improving End-to-End Contextual Speech Recognition with Fine-Grained Contextual Knowledge Selection https://ieeexplore.ieee.org/document/9747101

**Low-resource Speech Recognition**:
  - Efficiently Fusing Pretrained Acoustic and Linguistic Encoders for Low-resource Speech Recognition https://arxiv.org/abs/2101.06699

**Non-Autoregressive ASR**:
  - Boundary and Context Aware Training for CIF-based Non-Autoregressive End-to-end ASR https://arxiv.org/abs/2104.04702
  - A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation https://arxiv.org/abs/2110.05249
  
**Non-Autoregressive Lip Reading**:
  - Non-Autoregressive Lipreading Model with Integrate-and-Fire https://arxiv.org/abs/2008.02516

**Speech Translation**:
  - Exploring Continuous Integrate-and-Fire for Efficient and Adaptive Simultaneous Speech Translation https://arxiv.org/abs/2204.09595
  
**Spiking Neural Networks**:
  - Complex Dynamic Neurons Improved Spiking Transformer Network for Efficient Automatic Speech Recognition https://arxiv.org/abs/2302.01194

#### b. Repositories:

- A PyTorch implementation of a independent CIF module: https://github.com/MingLunHan/CIF-PyTorch

- CIF-based Contextualization, Collaborative Decoding (ColDec): https://github.com/MingLunHan/CIF-ColDec

- CIF as a bridge to connect pre-trained acoustic models and pre-trained language models: https://github.com/aispeech-lab/w2v-cif-bert

- This repository is the official implementation for the hierarchical knowledge distillation (HieraDist) developed for CIF-based models: https://github.com/MingLunHan/CIF-HieraDist
