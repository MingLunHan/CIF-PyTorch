# CIF-PyTorch
A PyTorch implementation of continuous integrate-and-fire (CIF) module for end-to-end (E2E) automatic speech recognition (ASR) [1].

## 1. Available Settings
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

## 2. Todo

- [ ] release a complete CIF-based speech recognizer 

## 3. References
[1] CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition https://arxiv.org/abs/1905.11235

[2] CIF-based Collaborative Decoding for End-to-end Contextual Speech Recognition https://arxiv.org/abs/2012.09466

[3] Improving End-to-End Contextual Speech Recognition with Fine-Grained Contextual Knowledge Selection https://arxiv.org/abs/2201.12806

[4] A Comparison of Label-Synchronous and Frame-Synchronous End-to-End Models for Speech Recognition https://arxiv.org/abs/2005.10113

[5] Efficiently Fusing Pretrained Acoustic and Linguistic Encoders for Low-resource Speech Recognition https://arxiv.org/pdf/2101.06699
