# Multi-Modal Analysis Pipeline

A comprehensive Python application for analyzing psychometric, vitals, and neurological imaging data. This application provides a robust pipeline for data ingestion, preprocessing, feature extraction, correlation analysis, and machine learning modeling.

## Features

- Multi-modal data processing support:
  - Eye tracking data
  - EEG data
  - Survey responses
  - Vital signs
  - Face heat maps
- Automated data preprocessing and cleaning
- Feature extraction from multiple data sources
- Correlation analysis and visualization
- Machine learning model training and evaluation
- Comprehensive logging and error handling
- Configurable pipeline parameters
- Production-ready code structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-modal-analysis.git
cd multi-modal-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. Install SpaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
project_root/
│
├── data/                    # Data directory
│   ├── eye_tracking/
│   ├── face_heatmap/
│   ├── vitals/
│   ├── eeg/
│   └── surveys/
│
├── src/                     # Source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── correlation_analysis.py
│   ├── machine_learning.py
│   ├── visualization.py
│   └── nlp_processing.py
│
├── tests/                   # Unit tests
│   └── test_*.py
│
├── output/                  # Generated outputs
│   ├── plots/
│   ├── results/
│   └── models/
│
├── config.py               # Configuration
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Usage

1. Prepare your data:
   - Place your data files in the appropriate subdirectories under `data/`
   - Ensure data files follow the expected format

2. Configure the pipeline:
   - Modify `config.py` to adjust parameters
   - Or create a custom configuration file

3. Run the pipeline:
```bash
# Basic usage
python main.py

# With debug mode
python main.py --debug

# With custom configuration
python main.py --config path/to/config.json
```

## Output

The pipeline generates several outputs in the `output/` directory:

- `plots/`: Visualization files
  - Correlation heatmaps
  - Feature distributions
  - Regression results
  - Time series plots

- `results/`: Analysis results
  - Feature matrices
  - Correlation analyses
  - Model performance metrics

- `models/`: Trained models
  - Saved model files
  - Model metadata

## Configuration

The pipeline can be configured through `config.py`. Key configuration options include:

- Data paths
- Analysis parameters
- Machine learning settings
- Output preferences

See `config.py` for detailed configuration options.

## Development

1. Set up development environment:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Format code:
```bash
black .
```

4. Run linting:
```bash
flake8 .
```

5. Run type checking:
```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
