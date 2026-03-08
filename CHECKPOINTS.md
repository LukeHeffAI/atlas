# Checkpoint Preparation

The checkpoints are available for download on [HuggingFace](https://huggingface.co/fredzzhang/atlas/tree/main).
Downloaded files should be extracted and organised in the following structure.

## Task Vector Checkpoints

```
└── atlas
    └── checkpoints
        └── ViT-B-32
            └── zeroshot_accuracies.json
            └── ft_accuracies.json
            └── head_CarsVal.pt
            └── CarsVal
                └── zeroshot.pt
                └── finetuned.pt
            └── head_DTDVal.pt
            └── DTDVal
                └── zeroshot.pt
                └── finetuned.pt
            └── ...
        └── ViT-B-16
            └── ...
        └── ViT-L-14
            └── ...
        └── RN50
            └── ...
        └── RN101
            └── ...
```

## Hypernetwork Checkpoints (Text-Based Adaptation)

After meta-training the text-to-coefficient hypernetwork, checkpoints are saved in the following structure:

```
└── atlas
    └── checkpoints
        └── ViT-B-32
            └── hypernetworks
                └── text_to_coef
                    └── meta_trained.pt     # Best validation accuracy model
                    └── meta_results.json   # Training history (losses, val accuracies)
            └── text_adapted
                └── <dataset>_hypernetwork.json  # Evaluation results
                └── <dataset>_synthetic.json     # Synthetic approach results
            └── ...
```

### Checkpoint Contents

**Task vector checkpoints** (`zeroshot.pt`, `finetuned.pt`):
- `model_state_dict`: Model weights

**Hypernetwork checkpoints** (`meta_trained.pt`):
- `model_state_dict`: Hypernetwork weights
- `model_config`: Hypernetwork configuration (architecture, dimensions, etc.)

### Loading Checkpoints

```python
# Load task vector
from src.task_vectors import NonLinearTaskVector
task_vector = NonLinearTaskVector(
    "checkpoints/ViT-B-32/CarsVal/zeroshot.pt",
    "checkpoints/ViT-B-32/CarsVal/finetuned.pt"
)

# Load hypernetwork
from src.hypernetworks.text_to_coef import TextToCoefHypernetwork
hypernetwork = TextToCoefHypernetwork.load(
    "checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt"
)
```