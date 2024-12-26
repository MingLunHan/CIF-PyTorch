## CIF-PyTorch

:rocket: **Attention! Please refer to https://github.com/MingLunHan/CIF-HieraDist for our latest and complete implementation of the CIF-based speech recognition model！**

A PyTorch implementation of Continuous Integrate-and-Fire (CIF) module for end-to-end (E2E) automatic speech recognition (ASR), which is originally proposed in **Cif: Continuous integrate-and-fire for end-to-end speech recognition** https://ieeexplore.ieee.org/document/9054250.

If you have any questions, please contact me through hanminglun1996@foxmail.com.

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

**LLM+CIF**
  - X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages https://github.com/phellonchen/X-LLM
  - Wav2Prompt: End-to-End Speech Prompt Generation and Tuning For LLM in Zero and Few-shot Learning https://arxiv.org/abs/2406.00522

**ASR**:
  - CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition https://ieeexplore.ieee.org/document/9054250 https://linhodong.github.io/cif_alignment/
  - A Comparison of Label-Synchronous and Frame-Synchronous End-to-End Models for Speech Recognition https://arxiv.org/abs/2005.10113
  - Knowledge Transfer from Pre-trained Language Models to CIF-based Speech Recognizers via Hierarchical Distillation https://www.isca-archive.org/interspeech_2023/han23_interspeech.pdf
  - CIF-T: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition https://arxiv.org/abs/2307.14132
  - CIF-RNNT: Streaming ASR Via Acoustic Word Embeddings with Continuous Integrate-and-Fire and RNN-Transducers https://ieeexplore.ieee.org/document/10448492
  - CIF-PT: Bridging Speech and Text Representations for Spoken Language Understanding via Continuous Integrate-and-Fire Pre-Training https://aclanthology.org/2023.findings-acl.566.pdf
  - A CIF-Based Speech Segmentation Method for Streaming E2E ASR https://ieeexplore.ieee.org/document/10081040
  - Improving CTC-Based Speech Recognition Via Knowledge Transferring from Pre-Trained Language Models https://ieeexplore.ieee.org/abstract/document/9747887
  - An efficient text augmentation approach for contextualized Mandarin speech recognition https://arxiv.org/abs/2406.09950

**ASR Context Biasing**:
  - CIF-based Collaborative Decoding for End-to-End Contextual Speech Recognition https://ieeexplore.ieee.org/document/9415054
  - Improving End-to-End Contextual Speech Recognition with Fine-Grained Contextual Knowledge Selection https://ieeexplore.ieee.org/document/9747101

**Low-resource Speech Recognition**:
  - Efficiently Fusing Pretrained Acoustic and Linguistic Encoders for Low-resource Speech Recognition https://arxiv.org/abs/2101.06699

**Non-Autoregressive ASR**:
  - Boundary and Context Aware Training for CIF-based Non-Autoregressive End-to-end ASR https://arxiv.org/abs/2104.04702
  - A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation https://arxiv.org/abs/2110.05249
  - Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition https://arxiv.org/abs/2206.08317
  - Paraformer-v2: An improved non-autoregressive transformer for noise-robust speech recognition https://arxiv.org/abs/2409.17746
  - E-Paraformer: A Faster and Better Parallel Transformer for Non-autoregressive End-to-End Mandarin Speech Recognition https://www.isca-archive.org/interspeech_2024/zou24_interspeech.pdf

**Non-Autoregressive Lip Reading**:
  - Non-Autoregressive Lipreading Model with Integrate-and-Fire https://arxiv.org/abs/2008.02516

**Speech Translation**:
  - UniST: Unified End-to-end Model for Streaming and Non-streaming Speech Translation https://www.semanticscholar.org/paper/UniST%3A-Unified-End-to-end-Model-for-Streaming-and-Dong-Zhu/e4d4728bd2e4ba9b91d4d57e98b5c81f84a88a5f
  - Exploring Continuous Integrate-and-Fire for Efficient and Adaptive Simultaneous Speech Translation https://arxiv.org/abs/2204.09595
  - Training Simultaneous Speech Translation with Robust and Random Wait-k-Tokens Strategy https://aclanthology.org/2023.emnlp-main.484/
  
**Spiking Neural Networks**:
  - Complex Dynamic Neurons Improved Spiking Transformer Network for Efficient Automatic Speech Recognition https://arxiv.org/abs/2302.01194

**Multimodal ASR**:
  - VILAS: Exploring the Effects of Vision and Language Context in Automatic Speech Recognition https://ieeexplore.ieee.org/document/10448450
  - X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages https://github.com/phellonchen/X-LLM
  - CM-CIF: Cross-Modal for Unaligned Modality Fusion with Continuous Integrate-and-Fire https://ieeexplore.ieee.org/abstract/document/9846612

**Keyword Spotting**
  - Leveraging Synthetic Speech for CIF-Based Customized Keyword Spotting https://link.springer.com/chapter/10.1007/978-981-97-0601-3_31

#### b. Repositories:
  - A PyTorch implementation of a independent CIF module: https://github.com/MingLunHan/CIF-PyTorch
  - A faster PyTorch implementation of CIF: https://github.com/George0828Zhang/torch_cif
  - CIF-based Contextualization, Collaborative Decoding (ColDec): https://github.com/MingLunHan/CIF-ColDec
  - CIF as a bridge to connect pre-trained acoustic models and pre-trained language models: https://github.com/aispeech-lab/w2v-cif-bert
  - The official implementation for the hierarchical knowledge distillation (HieraDist) developed for CIF-based models: https://github.com/MingLunHan/CIF-HieraDist
