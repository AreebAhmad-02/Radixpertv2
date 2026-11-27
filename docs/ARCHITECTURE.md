# Radixpert architecture and code map

This document explains how the code is organised and how data flows through it. For commands and options see
the [README](../README.md).

## 1. End-to-end flow

```
              ┌────────────────────────── training (scripts/train.py) ──────────────────────────┐
 ROCO v2 ─┐   │                                                                                   │
          ├─► data/datasets.py ─► MultiDatasetLoader (per stage) ─► training/multi_stage_trainer │
 PadChest ┘   │                                                         │                         │
              │                                                         ▼                         │
              │                                     models/radixpert.py  (Radixpert)             │
              │   image ─► VisionEncoder ─┐                                                       │
              │                           ├─► HierarchicalCrossModalFusion ─► Llama (+MSA-LoRA)   │
              │   text  ─► token embeds ──┘                                        │              │
              │                                                                    ▼              │
              │                                                          report text / logits     │
              └───────────────────────────────────────────────────────────────────────────────────┘
 checkpoints (.pt) ─► scripts/evaluate.py (metrics)  and  scripts/inference.py (reports)
```

## 2. Training stages

`MultiStageTrainer.train_all_stages` runs stages in order. Each stage has its own data, learning rate, LoRA rank
and loss weights.

| Stage | Enum (`TrainingStage`) | Data | Frozen / trainable |
| --- | --- | --- | --- |
| 1 | `PRE_ALIGNMENT` | ROCO v2 | Llama frozen; LoRA, vision encoder, HCF trainable |
| 2 | `DOMAIN_ADAPTATION` | ROCO v2 + PadChest | Llama frozen; LoRA, vision encoder, HCF trainable |
| 3 | `TASK_SPECIFIC` | ROCO v2 + PadChest | Llama unfrozen; everything trainable |

The freezing rules are in `Radixpert._configure_training_stage`. A checkpoint `stage_<n>_final.pt` is written after
each stage.

## 3. Module reference

### `models/`

| File | Main classes / functions | What it does |
| --- | --- | --- |
| `radixpert.py` | `RadixpertConfig`, `Radixpert`, `create_radixpert_model` | The full model. Loads Llama, builds the vision encoder and HCF module, wraps the target linear layers with MSA-LoRA, adds medical special tokens, and exposes `forward`, `generate_report`, `get_trainable_parameters` and stage checkpoint save/load. |
| `msa_lora.py` | `MSALoRAConfig`, `MSALoRALayer`, `MSALoRAManager` | Stage-dependent LoRA: rank schedule, per-stage scaling, batch-norm (stage 2) and learnable gate (stage 3), L2 and orthogonality regularisation, in-place rank resizing when the stage changes. |
| `hcf_fusion.py` | `HCFConfig`, `CrossModalAttention`, `AdaptiveGatingMechanism`, `HierarchicalFusionLayer`, `HierarchicalCrossModalFusion` | Hierarchical cross-modal fusion: project both modalities to a shared space, run several fusion levels with cross-attention and gating, merge the levels with an MLP, and project back to text and vision dimensions. Includes `vision_only_forward`, `get_attention_maps` and `compute_fusion_loss`. |
| `vision_encoder.py` | `VisionEncoderConfig`, `MedicalImagePreprocessor`, `MultiScaleFeatureExtractor`, `SpatialAttentionModule`, `ChannelAttentionModule`, `PositionalEncoding2D`, `VisionEncoder` | Image branch: selectable backbone (`efficientnet_b7`, `resnet50`, `densenet121`, `vit_base`, `swin_transformer`, via timm), multi-scale features, attention modules, positional encoding, progressive unfreezing (`unfreeze_backbone_progressively`). |

### `training/`

| File | Main classes / functions | What it does |
| --- | --- | --- |
| `multi_stage_trainer.py` | `StageConfig`, `MultiStageTrainer` | Stage loop, per-stage optimizer and scheduler, mixed precision, validation, checkpoint save/load, final evaluation. |
| `losses.py` | `LossConfig`, `GenerationLoss`, `ContrastiveLoss`, `ClinicalValidationLoss`, `FusionLoss`, `MSALoRARegularizationLoss`, `FocalLoss`, `MultiStageLoss` | Individual loss terms and `MultiStageLoss.compute_loss`, which combines them using the per-stage weights in `LossConfig.stage_weights`. |
| `optimization.py` | `OptimizationConfig`, `LayerWiseLearningRateOptimizer`, `MedicalOptimizer`, `SAM`, `Lookahead`, LR schedulers | Optimizer construction (AdamW and variants), layer-wise learning rates, warm-up schedulers, gradient clipping helpers. |

### `data/`

| File | Main classes / functions | What it does |
| --- | --- | --- |
| `datasets.py` | `DatasetConfig`, `BaseRadiologyDataset`, `PadChestDataset`, `ROCOv2Dataset`, `MultiDatasetLoader`, `create_*_config`, `create_multi_stage_datasets` | Reads annotations, filters samples, makes the 80/10/10 split, loads images, builds the text string, and produces batches. `MultiDatasetLoader` picks the datasets for a given stage. |
| `preprocessing.py` | `MedicalImagePreprocessor`, `TextPreprocessor` | Image intensity normalisation, CLAHE, contrast and noise handling; report cleaning, abbreviation normalisation, Spanish keyword translation, section markers. |
| `transforms.py` | `MedicalAugmentation`, `NoiseReduction`, `get_medical_transforms` | Training-time augmentation and evaluation transforms. |

### `evaluation/`

| File | Main classes / functions | What it does |
| --- | --- | --- |
| `metrics.py` | `BLEUMetric`, `ROUGEMetric`, `CIDErMetric`, `METEORMetric`, `BERTScoreMetric`, `RadGraphF1Metric`, `ClinicalAccuracyMetric`, `RadCliQMetric`, `RadixpertEvaluator` | Text-generation and clinical-content metrics. `RadixpertEvaluator.evaluate` runs the model over a dataloader and returns all metrics; `print_evaluation_report` formats them. |

### `scripts/`

| File | What it does |
| --- | --- |
| `prepare_data.py` | `DataPreparationPipeline`: validates images, filters by report length, writes processed annotation files and statistics for PadChest and ROCO v2. The command-line parser is not finished yet. |
| `train.py` | Parses arguments, builds configs, datasets and the trainer, runs all stages, then runs a final evaluation and writes `final_evaluation_metrics.json`. |
| `evaluate.py` | Loads a checkpoint and evaluates it on a chosen dataset and split; optional prediction dumps and attention visualisations. |
| `inference.py` | `RadixpertInference` class and CLI: single-image and batch report generation, optional formatted clinical report output. |

### `utils/`

| File | What it does |
| --- | --- |
| `checkpoint_utils.py` | `CheckpointManager` (save, load, best and latest checkpoint lookup, history, cleanup) and `find_latest_checkpoint`. |
| `device_utils.py` | Device selection, memory reporting and clean-up, a simple `DeviceMonitor`. |
| `logging_utils.py` | `setup_comprehensive_logging`, system and model information logging. |
| `visualization.py` | `RadixpertVisualizer`: attention maps, feature maps, training curves, report analysis plots. |

## 4. Tensors and shapes (defaults)

| Tensor | Shape / size |
| --- | --- |
| Input image | `[B, 3, 512, 512]` (resized by the data pipeline) |
| Vision feature dimension | 1024 |
| Text / fusion dimension | 4096 |
| Fusion levels / heads | 3 / 16 |
| Max text length | 512 tokens (training), 2048 (model context setting) |

## 5. Extending the code

- **New dataset:** subclass `BaseRadiologyDataset` and implement `_load_data`, `_get_image_path`,
  `_get_text_content` and, if needed, `_passes_quality_check`; then add it to `MultiDatasetLoader`.
- **New vision backbone:** add it to `VisionEncoder._create_backbone` and to the `--vision_backbone` choices in
  `scripts/train.py`.
- **New loss term:** add a module in `training/losses.py`, a weight in `LossConfig.stage_weights`, and a call in
  `MultiStageLoss.compute_loss`.
- **Different LoRA targets:** change `target_modules` in `RadixpertConfig`.
