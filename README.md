# Paper Implementations

A growing collection of deep learning research paper implementations, primarily built with PyTorch.

The goal of this repository is to understand how modern deep learning models work by translating ideas from research papers into clear, modular code. Each implementation is an opportunity to study the model architecture, follow the data flow through its components, and connect the paper's concepts to a working implementation.

## What you'll find here

- Research-driven implementations of deep learning architectures and methods.
- PyTorch-based code designed to make model components easier to inspect and understand.
- Hands-on exploration of the design choices behind research models, from architecture to training components.

This repository is a work in progress. New implementations will be added as I study and implement additional papers.

## Implementation principles

- **Readable:** model components are separated so the architecture is easier to follow.
- **Faithful to the paper:** design choices are guided by the original publication, while practical deviations should be documented.
- **Understandable:** the code is intended to support learning, not merely provide a black-box model.
- **Reproducible:** paper-specific setup, training details, and evaluation results should be documented when available.

Not every implementation is a full reproduction of the paper's reported results. Please refer to the individual implementation for its current scope and status.

## Tech stack

- Python
- PyTorch

Additional libraries may vary by implementation.

## Implemented Papers
 
| Status | Paper | Authors | Year | Task | Code |
|:------:|-------|---------|:----:|------|:----:|
| ✅ | [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton | 2012 | Image Classification | [AlexNet](./AlexNet) |
| ✅ | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | Ronneberger, Fischer, Brox | 2015 | Semantic Segmentation | [U-Net](./U-Net) |
| ✅ | [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) | He, Zhang, Ren, Sun | 2015 | Image Classification | [ResNet](./ResNet) |
| ✅ | [You Only Look Once: Unified, Real-Time Object Detection (YOLOv1)](https://arxiv.org/abs/1506.02640) | Redmon, Divvala, Girshick, Farhadi | 2015 | Object Detection | [yolo-v1](./yolo-v1) |
| ✅ | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929) | Dosovitskiy et al. | 2020 | Image Classification | [ViT](./ViT) |
| ✅ | [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | Radford et al. | 2021 | Vision-Language Learning | [CLIP](./CLIP) |
 
### Up Next
 
| Status | Paper | Task |
|:------:|-------|------|
| 🔜 | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | Semantic Segmentation |
 
**Legend:** ✅ Implemented · 🚧 In progress · 🔜 Planned

## Author

**Mahmoud Etman**
- GitHub: https://github.com/ma74a
- LinkedIn: https://www.linkedin.com/in/mahm0ud-etman/