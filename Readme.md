## Training Cheat Sheet

```bash
python training/train.py --model MODEL --method METHOD --data DATA_DIR [OPTIONS]
```

### Required

| Flag           | Description                                                        |
|----------------|--------------------------------------------------------------------|
| `--model`      | Backbone model: `mobilenet`, `efficientnet`                        |
| `--method`     | Training method: `triplet`                                         |
| `--data`       | Path to dataset directory (e.g., `data/hot_wheels`)                |

### Optional

| Flag             | Description                                                                 | Default             |
|------------------|-----------------------------------------------------------------------------|---------------------|
| `--view_mode`     | Set to `multi` for multi-view triplet training                            | `multi`             |
| `--anchor_mode`   | `multi` (fused anchor) or `single` (1-shot reference image anchor)         | `multi`             |
| `--embedding_dim` | Size of the output embedding vector                                         | 128                 |
| `--epochs`        | Number of training epochs                                                   | 20                  |
| `--batch_size`    | Batch size                                                                  | 32                  |
| `--lr`            | Learning rate                                                               | 1e-4                |
| `--save`          | Path to save final trained model                                            | `trained_model.pt`  |
| `--resume`        | Resume training from an existing checkpoint                                 | None                |
| `--no_parallel`   | Disable multi-GPU training (DataParallel)                                   | Off                 |
| `--patience`      | Early stopping patience in number of epochs                                 | 5                   |
| `--log`           | CSV path for saving training logs                                            | `train_log.csv`     |

### Example Commands

Basic multi-view training:
```bash
python3 training/train.py \
  --model mobilenet \
  --method triplet \
  --data data/hot_wheels \
  --view_mode multi \
  --anchor_mode multi \
  --save checkpoints/mobilenet_multiview.pt
```

Training with one-shot reference anchor:
```bash
python3 training/train.py \
  --model efficientnet \
  --method triplet \
  --data data/hot_wheels \
  --view_mode multi \
  --anchor_mode single \
  --save checkpoints/efficientnet_1shot.pt \
  --no_parallel \
  --patience 8
```

Example
```bash
python3 training/train.py \
  --model swifttracknet \
  --method triplet \
  --data data/hot_wheels \
  --view_mode multi \
  --resume checkpoints/swifttracknet_mutliview_v1.pt \
  --save checkpoints/swifttracknet_mutliview_v1.pt \
  --no_parallel
```

---

## Testing Cheat Sheet

```bash
python3 test_matchnet.py \
  --model MODEL \
  --checkpoint PATH_TO_PT \
  --query QUERY_FOLDER \
  --gallery GALLERY_ROOT \
  [--topk N] [--visualize]
```

### Required

| Flag            | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| `--model`        | Model type: `lightmatchnet` or `efficientnet`                             |
| `--checkpoint`   | Path to trained weights `.pt` file                                        |
| `--query`        | Folder containing `_02.jpg` reference views (e.g., `id_003/left_02.jpg`)  |
| `--gallery`      | Folder containing ID subfolders with `_01.jpg` views for matching         |

### Optional

| Flag           | Description                             | Default |
|----------------|-----------------------------------------|---------|
| `--topk`        | Number of top matches to return         | 5       |
| `--visualize`   | Show query and top-k matched images     | False   |

### Example Command

```bash
python3 test_matchnet.py \
  --model lightmatchnet \
  --checkpoint checkpoints/mobilenet_mutliview_v1.pt \
  --query data/hot_wheels/id_005 \
  --gallery data/hot_wheels \
  --topk 5 \
  --visualize
```

```bash
python evaluate_fusion.py \
  --model swifttracknet \
  --checkpoint checkpoints/swifttracknet.pt \
  --query veri_dataset/sample_query/0001 \
  --gallery veri_dataset/sample_gallery \
  --topk 5 \
  --visualize \
  --output_csv fused_results_swift.csv
```