# Genomic Data Science - ML Project Template

A comprehensive PyTorch-based machine learning project template for genomic data science and bioinformatics applications. This template provides a complete structure with everything needed to start developing ML models for scientific purposes.

## 🚀 Features

- **Ready-to-use PyTorch setup** with modular architecture
- **Data loading utilities** for genomic data
- **Configurable training pipeline** with YAML config files
- **Comprehensive logging** with TensorBoard support
- **Model evaluation** and inference scripts
- **Unit tests** with pytest
- **Jupyter notebooks** for exploration
- **Modern Python packaging** with setuptools
- **Scientific computing stack** (NumPy, Pandas, SciPy, Biopython)

## 📁 Project Structure

```
Genomic-Data-Science/
├── configs/                    # Configuration files
│   └── default_config.yaml
├── data/                       # Data directory
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── logs/                       # Training logs
├── models/                     # Saved models
│   └── checkpoints/            # Model checkpoints
├── notebooks/                  # Jupyter notebooks
│   └── exploratory_analysis.ipynb
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset class
│   │   └── dataloader.py       # DataLoader utilities
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   └── base_model.py       # Base model template
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── logger.py           # Logging utilities
│       └── metrics.py          # Metrics calculation
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_utils.py
├── train.py                    # Training script
├── evaluate.py                 # Evaluation/inference script
├── setup.py                    # Package setup
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🔧 Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/JulianPMenon/Genomic-Data-Science.git
cd Genomic-Data-Science

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/JulianPMenon/Genomic-Data-Science.git
cd Genomic-Data-Science

# Create conda environment
conda env create -f environment.yml
conda activate genomic-ml

# Install package in development mode
pip install -e .
```

## 🎯 Quick Start

### 1. Training a Model

```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py --config configs/custom_config.yaml
```

### 2. Evaluating a Model

```bash
# Evaluate on test data
python evaluate.py \
    --checkpoint models/checkpoints/best_model.pth \
    --data-path data/test.csv \
    --mode evaluate

# Run inference on new data
python evaluate.py \
    --checkpoint models/checkpoints/best_model.pth \
    --data-path data/new_data.csv \
    --mode inference \
    --output predictions.npz
```

### 3. Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/exploratory_analysis.ipynb
```

## 📊 Data Format

The default dataset expects data in CSV format with features in columns and labels in the last column. You can customize the `GenomicDataset` class in `src/data/dataset.py` for different data formats.

Example CSV structure:
```
feature1,feature2,feature3,...,featureN,label
0.123,0.456,0.789,...,0.321,0
0.234,0.567,0.890,...,0.432,1
...
```

## ⚙️ Configuration

Edit `configs/default_config.yaml` to customize:

- **Model architecture** (input_dim, hidden_dim, output_dim, dropout)
- **Training parameters** (batch_size, learning_rate, epochs, optimizer)
- **Data splits** (train/val/test ratios)
- **Logging settings** (TensorBoard, checkpoints)
- **Device settings** (CPU/GPU, mixed precision)

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/
```

## 📝 Customization Guide

### Adding a New Model

1. Create a new file in `src/models/` (e.g., `cnn_model.py`)
2. Inherit from `nn.Module` and implement `forward()`
3. Import it in `src/models/__init__.py`
4. Update `train.py` to use your model

### Custom Data Loading

1. Modify `src/data/dataset.py` to handle your data format
2. Update `_load_from_file()` method for custom parsing
3. Adjust `__getitem__()` for data transformations

### Adding New Metrics

1. Add metric functions to `src/utils/metrics.py`
2. Update `calculate_metrics()` to include new metrics
3. Modify `print_metrics()` for display

## 🔬 Scientific Use Cases

This template is suitable for:

- **Gene expression analysis** (RNA-seq, microarray)
- **DNA/RNA sequence classification**
- **Protein structure prediction**
- **Disease classification** from genomic markers
- **Drug response prediction**
- **Variant effect prediction**
- **Multi-omics data integration**

## 📚 Dependencies

### Core ML Libraries
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- scikit-learn >= 1.3.0

### Scientific Computing
- SciPy >= 1.10.0
- Biopython >= 1.81

### Visualization
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Plotly >= 5.14.0

### Development Tools
- pytest >= 7.3.0
- black >= 23.3.0
- flake8 >= 6.0.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Julian P. Menon
- GitHub: [@JulianPMenon](https://github.com/JulianPMenon)

## 🙏 Acknowledgments

This template is designed for scientific research and educational purposes in genomic data science.

## 📖 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Biopython Tutorial](http://biopython.org/DIST/docs/tutorial/Tutorial.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 🐛 Issues and Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.
