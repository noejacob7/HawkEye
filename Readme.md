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

```bash
python3 training/train.py   --model swifttracknet   --method triplet   --data data/VeRi/image_train   --label data/VeRi/train_label.xml   --dataset_type veri   --view_mode multi   --anchor_mode multi   --embedding_dim 128   --epochs 30   --batch_size 16   --lr 1e-4   --save checkpoints/swifttracknet_multiview_v2.pt --no_parallel
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
python3 evaluate_veri.py   --model mobilenet   --checkpoint checkpoints/mobilenet_multiview_v1.pt   --query_dir data/VeRi/image_query   --query_label data/VeRi/test_label.xml   --gallery_dir data/VeRi/image_test   --gallery_label data/VeRi/test_label.xml   --output_csv results/veri_mobilenet.csv   --topk 10
```

# Evaluation Metrics

| Category | Metric | Why it Matters |
|:---|:---|:---|
| **Matching Quality** | Top-1 / Top-5 / Top-10 accuracy | Direct matching ability |
| | Mean Average Precision (mAP) | Ranking quality |
| | Correct / Incorrect match ratio | Raw success vs fail |
| **Network Efficiency** | Number of parameters | Model size for storage |
| | Model file size (MB or KB) | Memory footprint |
| | Number of layers/blocks | Simplicity, depth |
| | FLOPs (floating point operations) | How heavy the computation is |
| **Speed / Deployment** | Inference time per image | How fast per prediction |
| | FPS (Frames Per Second) | Real-time capability |
| | Total evaluation time (whole query set) | Batch speed at scale |
| | CUDA memory usage (if GPU) | RAM footprint at runtime |


| Metric | Code ready? |
|:---|:---|
| Top-1 / Top-5 / Top-10 / mAP | Matching code |
| Correct / Incorrect counts | Simple counter |
| Parameters | Code ready |
| File size | Code ready |
| FLOPs | `ptflops` method |
| Inference time | Code ready |
| FPS | Code ready |
| CMC Curve (optional) |  later if needed |


# Results

## Training on hotwheels -- Testing on veri

### Swifttracknet

Top-1 Accuracy: 63.50%
Top-5 Accuracy: 86.00%
Top-10 Accuracy: 91.00%
Total Queries: 200
Correct Matches: 200
Incorrect Matches: 0
Avg Query Inference Time: 0.010043 sec
Avg Gallery Inference Time: 0.057173 sec

### Efficientnet

Top-1 Accuracy: 36.50%
Top-5 Accuracy: 60.00%
Top-10 Accuracy: 70.50%
Total Queries: 200
Correct Matches: 200
Incorrect Matches: 0
Avg Query Inference Time: 0.055033 sec
Avg Gallery Inference Time: 0.362899 sec

### Mobilenet

Top-1 Accuracy: 93.00%
Top-5 Accuracy: 99.50%
Top-10 Accuracy: 100.00%
Total Queries: 200
Correct Matches: 200
Incorrect Matches: 0
Avg Query Inference Time: 0.035187 sec
Avg Gallery Inference Time: 0.226596 sec

### Training on veri -- Testing on veri

### Mobilenet

Top-1 Accuracy: 94.50%
Top-5 Accuracy: 100.00%
Top-10 Accuracy: 100.00%
Total Queries: 200
Correct Matches: 200
Incorrect Matches: 0
Avg Query Inference Time: 0.042103 sec
Avg Gallery Inference Time: 0.264792 sec

### Swifttracknet

Top-1 Accuracy: 90.50%
Top-5 Accuracy: 98.50%
Top-10 Accuracy: 99.50%
Total Queries: 200
Correct Matches: 199
Incorrect Matches: 1
Avg Query Inference Time: 0.012529 sec
Avg Gallery Inference Time: 0.065119 sec

Top-1 Accuracy: 93.50%
Top-5 Accuracy: 98.50%
Top-10 Accuracy: 99.50%
Total Queries: 200
Correct Matches: 199
Incorrect Matches: 1
Avg Query Inference Time: 0.011895 sec
Avg Gallery Inference Time: 0.063725 sec