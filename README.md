# SwiftXPlain

This repository provides an implementation of several algorithms for computing 
distance-restricted 𝜖<sub>∞</sub>-ball abductive explanations (AXp), contrastive explanations (CXp), smallest constrstive explanations 
or for listing partial/complete explanations for deep neural networks (DNNs).

## 🚀 Running the Tool

### General Usage
```
python3 ./robxpl.py -x 1 --eps <epsilon-ball> -c <NUM_CPU/GPU> --alg <algorithm> -o <robustness-oracle> <model .onnx>  <input-batch .npy> 
```

##### Print options:
```
python3 ./robxpl.py --help 
```

##### Linear-search algorithm:
```
python3 ./robxpl.py -x 1 --eps 0.03 -c 1 --alg linear -o crown ../tests/onnx/mnist/MNIST_dense.onnx   ./tests/data/MNIST/inst_dense.npy 
```

##### Dichotomic search algorithm:
```
python3 ./robxpl.py -x 1 --eps 0.03 -c 1 --alg dicho -o crown ../tests/onnx/mnist/MNIST_dense.onnx   ./tests/data/MNIST/inst_dense.npy 
```

##### SwiftXplain algorithm (with Multi-GPUs):
```
python3 ./robxpl.py -x 1 --eps 0.03 -c 16 --alg swift --FD 0.85 -o crown ../tests/onnx/mnist/MNIST_dense.onnx   ./tests/data/MNIST/inst_dense.npy 
```

## Citations

Please cite the following papers when you use this work:

```
@inproceedings{swiftxp-kr24,
    title     = {{Distance-Restricted Explanations: Theoretical Underpinnings \& Efficient Implementation}},
    author    = {Izza, Yacine and Huang, Xuanxiang and Morgado, Antonio and Planes, Jordi and Ignatiev, Alexey and Marques-Silva, Joao},
    OPTbooktitle = {{Proceedings of the 21st International Conference on Principles of Knowledge Representation and Reasoning}},
    booktitle = {KR},
    pages     = {475--486},
    year      = {2024},
    OPTdoi       = {10.24963/kr.2024/45},
    OPTurl       = {https://doi.org/10.24963/kr.2024/45},
  }

@article{ims-corr24c,
  author       = {Yacine Izza and
                  Joao Marques{-}Silva},
  title        = {Efficient Contrastive Explanations on Demand},
  journal      = {CoRR},
  volume       = {abs/2412.18262},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2412.18262},
  doi          = {10.48550/ARXIV.2412.18262}
}

@article{hms-corr23,
  author       = {Xuanxiang Huang and
                  Jo{\~{a}}o Marques{-}Silva},
  title        = {From Robustness to Explainability and Back Again},
  journal      = {CoRR},
  volume       = {abs/2306.03048},
  year         = {2023},
  OPTurl          = {https://doi.org/10.48550/arXiv.2306.03048},
  OPTdoi          = {10.48550/ARXIV.2306.03048},
  OPTeprinttype    = {arXiv},
  OPTeprint       = {2306.03048},
  timestamp    = {Tue, 13 Jun 2023 15:56:49 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2306-03048.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
