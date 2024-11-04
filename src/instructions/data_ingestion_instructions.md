# Data Ingestion Module Instructions

## Overview

The `data_ingestion.py` module provides a graphical user interface (GUI) for loading, preprocessing, and analyzing various types of data including eye tracking, face heatmap, vitals, EEG, survey, and JSON data.

## Module Structure

### Main Classes

#### DataIngestionGUI

The main GUI class that handles the interface and user interactions. Key components:

- Main window with horizontal paned layout
- Left frame containing data loading buttons
- Right pane split vertically into status and preview sections
- Analysis and export option frames

### Key Functions

#### Data Loading Functions

- `load_eye_tracking_data(path)`: Loads eye tracking CSV data with timestamp and position columns
- `load_face_heatmap_data(path)`: Loads face heatmap CSV data with timestamp and heatmap values
- `load_vitals_data(path)`: Loads vitals CSV data with timestamp, heart rate, blood pressure, temperature
- `load_eeg_data(path)`: Loads EEG data from EDF files using MNE library
- `load_survey_data(path)`: Loads survey CSV data with participant ID, questions and responses
- `load_json_data(path)`: Loads arbitrary JSON data structures

Each loading function:

- Validates file existence and format
- Checks for required columns/structure
- Logs success/failure
- Returns appropriate data structure

#### Data Processing Methods

- `preprocess_data()`: Handles data cleaning and preparation
- `extract_features()`: Extracts relevant features from data
- `analyze_correlations()`: Performs correlation analysis between data types

#### Export Methods  

- `export_to_csv()`: Exports processed data to CSV files
- `export_to_pickle()`: Exports all data structures to pickle format

### Directory Structure

The module creates and uses the following directory structure:
