This is a unofficial PyTorch implementation of EdgeViT in "EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers", arXiv 2022.



Pretrained models will come soon.



## Usage

```python
from edgevit import EdgeViT_XXS, EdgeViT_XS, EdgeViT_S

model = EdgeViT_XXS()
inputs = torch.randn((1, 3, 224, 224))
print(model(inputs))
```



## Citation

```
@article{pan2022edgevits,
  title={EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers},
  author={Pan, Junting and Bulat, Adrian and Tan, Fuwen and Zhu, Xiatian and Dudziak, Lukasz and Li, Hongsheng and Tzimiropoulos, Georgios and Martinez, Brais},
  journal={arXiv preprint arXiv:2205.03436},
  year={2022}
}
```



