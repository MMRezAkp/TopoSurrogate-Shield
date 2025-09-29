# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added
- Initial release of Neural Network Activation Extractor
- Support for multiple architectures (ResNet-18/34/152, MobileNetV2, EfficientNet-B0)
- Multiple tap modes for activation extraction:
  - `topotroj_compat`: TopoTroj compatible mode
  - `toap_block_out`: TOAP block output mode
  - `bn2_legacy`: BatchNorm2d legacy mode
  - `spatial_preserve`: Spatial structure preservation mode
- Topological analysis capabilities:
  - Correlation matrix computation
  - Persistent homology computation
  - Topological Summary Statistics (TSS)
- Backdoor detection support with triggered inputs
- Performance monitoring and complexity analysis
- Memory optimization for large-scale processing
- Comprehensive logging and timing tracking
- Automatic architecture detection from model state dict
- Batch processing with configurable batch sizes
- Support for CIFAR-10 and CIFAR-100 datasets
- Metadata collection and export
- JSON and CSV output formats
- Command-line interface with extensive options

### Features
- **Architecture Support**: Auto-detection and manual specification
- **Flexible Extraction**: Multiple hook strategies for different analysis needs
- **Scalability**: Memory-efficient processing of large datasets
- **Analysis Pipeline**: End-to-end topological analysis workflow
- **Reproducibility**: Deterministic seeding and consistent results
- **Monitoring**: Real-time memory and performance tracking

### Documentation
- Comprehensive README with usage examples
- MIT License included
- Requirements specification
- Changelog documentation
- Code documentation and comments

### Dependencies
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- tqdm >= 4.62.0
- psutil >= 5.8.0
