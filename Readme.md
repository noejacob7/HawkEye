## Training Cheat Sheet

```bash
python training/train.py --model MODEL --method METHOD --data DATA_DIR [OPTIONS]
```

### Required

| Flag         | Description                                         |
|--------------|-----------------------------------------------------|
| `--model`    | Model architecture (`lightmatchnet`, `efficientnet`, `ghost`, `siamese`) |
| `--method`   | Training approach (`triplet`, `siamese`, `contrastive`) |
| `--data`     | Path to root folder of dataset (e.g., `lightmatch_data`) |

### Optional

| Flag         | Description                                         | Default             |
|--------------|-----------------------------------------------------|---------------------|
| `--epochs`   | Number of epochs                                    | 20                  |
| `--batch_size`| Number of samples per batch                        | 32                  |
| `--lr`       | Learning rate                                       | 1e-4                |
| `--save`     | Save model to path                                  | `trained_model.pt`  |
| `--resume`   | Resume from existing checkpoint                     | None                |
