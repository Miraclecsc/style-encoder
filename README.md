# Style Encoder

A cross-modal style extraction framework via Textual Inversion for establishing transferable style representations through staged training.

## Overview

This project proposes a novel approach to cross-modal style extraction using Textual Inversion techniques. The framework establishes transferable style representations through a multi-stage training pipeline, enabling high-quality style transfer and generation tasks.

### Key Contributions

- **Cross-modal Style Extraction**: Leverages Textual Inversion to extract and encode visual style attributes into text embeddings, creating a unified representation across modalities.
- **Staged Training Pipeline**: Implements a systematic multi-stage training approach for optimal style embedding learning.
- **Expanded Style30k Dataset**: Extended the Style30k dataset with detailed captions, creating comprehensive image-text pairs specifically designed for style embedding training.

## Project Structure

```
style-encoder/
├── train/              # Training scripts
│   ├── train_emb.py    # Embedding training
│   ├── train_enc.py    # Encoder training
│   ├── train_ftn.py    # Fine-tuning training
│   └── train_new.py    # New training pipeline
├── inference/          # Inference and testing scripts
│   ├── test_emb.py     # Embedding testing
│   ├── test_enc.py     # Encoder testing
│   └── test_enc_new.py # New encoder testing
├── src/                # Core model implementations
│   ├── pipeline_stable_diffusion.py
│   ├── unet_2d_blocks.py
│   └── unet_2d_condition.py
├── my_clip/            # Custom CLIP model implementation
│   ├── __init__.py
│   └── modeling_clip.py
├── utils/              # Utility scripts and tools
├── requirements.txt    # Python dependencies
└── environment.yml     # Conda environment configuration
```

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate style
```

### Using pip

```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.10+
- PyTorch 2.6.0
- Transformers 4.44.2
- Diffusers 0.30.2
- CLIP
- Accelerate
- Other dependencies listed in `requirements.txt`

## Usage

### Training

The training pipeline consists of multiple stages:

1. **Embedding Training**: Train style embeddings using the expanded Style30k dataset
   ```bash
   python train/train_emb.py
   ```

2. **Encoder Training**: Train the style encoder network
   ```bash
   python train/train_enc.py
   ```

3. **Fine-tuning**: Fine-tune the complete model
   ```bash
   python train/train_ftn.py
   ```

### Inference

Run inference using trained models:

```bash
python inference/test_emb.py
```

Or test the encoder:

```bash
python inference/test_enc.py
```

## Dataset

This project utilizes the **expanded Style30k dataset**, which includes:
- 30,000+ stylized images
- Detailed image-text pairs for each style
- Comprehensive captions designed for embedding training
- Multiple style categories and attributes

## Methodology

### Cross-modal Style Extraction via Textual Inversion

The framework employs Textual Inversion to learn transferable style representations:

1. **Style Encoding**: Extract visual style features from images
2. **Text Embedding Mapping**: Map style features to the CLIP text embedding space
3. **Joint Training**: Optimize both visual and textual representations simultaneously
4. **Style Transfer**: Apply learned embeddings for style generation tasks

### Staged Training Approach

1. **Stage 1**: Initial embedding learning with Style30k image-text pairs
2. **Stage 2**: Encoder training for robust style feature extraction
3. **Stage 3**: End-to-end fine-tuning for optimal performance

## Model Architecture

The framework integrates several key components:
- **CLIP Vision Encoder**: Extract visual features
- **CLIP Text Encoder**: Process text embeddings
- **Custom Style Encoder**: Map between modalities
- **Stable Diffusion Pipeline**: Generate styled images

## Citation

If you use this code or the expanded Style30k dataset in your research, please cite:

```bibtex
@article{style-encoder,
  title={Cross-modal Style Extraction via Textual Inversion},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

[License information to be added]

## Acknowledgments

- Style30k dataset creators
- CLIP and Stable Diffusion teams
- Hugging Face Transformers and Diffusers libraries
