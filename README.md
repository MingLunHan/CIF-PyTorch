## CIF-PyTorch
A PyTorch implementation of continuous integrate-and-fire (CIF) module for end-to-end (E2E) automatic speech recognition (ASR) [1].

### 1. Available Settings
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

### 2. Todo

- [ ] release a complete CIF-based speech recognizer 

### 3. **Other CIF Resources**

#### a. Papers:

**ASR**:
  - CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition https://ieeexplore.ieee.org/document/9054250
  - A Comparison of Label-Synchronous and Frame-Synchronous End-to-End Models for Speech Recognition https://arxiv.org/abs/2005.10113

**ASR Contextualization & Customization & Personalization**:
  - CIF-based Collaborative Decoding for End-to-End Contextual Speech Recognition https://ieeexplore.ieee.org/document/9415054
  - Improving End-to-End Contextual Speech Recognition with Fine-Grained Contextual Knowledge Selection https://arxiv.org/abs/2201.12806

**Low-resource Speech Recognition**:
  - Efficiently Fusing Pretrained Acoustic and Linguistic Encoders for Low-resource Speech Recognition https://arxiv.org/abs/2101.06699

**Non-autoregressive ASR**:
  - Boundary and Context Aware Training for CIF-based Non-Autoregressive End-to-end ASR https://arxiv.org/abs/2104.04702
  - A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation https://arxiv.org/abs/2110.05249

#### b. Repositories:

- A PyTorch implementation of a independent CIF module: https://github.com/MingLunHan/CIF-PyTorch

- CIF-based Contextualization: https://github.com/MingLunHan/CIF-ColDec
