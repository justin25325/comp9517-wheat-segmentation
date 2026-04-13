## EWS masks + shared loader

### Why
EWS `*_mask.png` files are **2-channel** and not strictly binary. We verified the correct foreground label by generating overlay images; the initial assumption (`mask == 255` as plant) was inverted.

### Ground truth rule (locked in)
Use **channel 0** and binarise as:
- **Plant (foreground) = 1 where `mask_ch0 == 0`**
- Background = 0 otherwise

### How to use
Use the shared loader in `src/datasets/ews.py` so all methods use identical labels.

```python
from src.datasets.ews import list_pairs, load_sample

pairs = list_pairs("data/EWS-Dataset", "train")
img, gt = load_sample(pairs[0])
