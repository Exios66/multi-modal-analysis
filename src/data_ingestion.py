import os
import pandas as pd
import mne
import json
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Union, Optional, Any
import pickle
import numpy as np
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directories
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "results" 
INGESTION_DIR = OUTPUT_DIR / "ingestion"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, RESULTS_DIR, INGESTION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class DataIngestionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Ingestion Interface")
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create horizontal paned window
        self.h_paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.h_paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left pane for buttons
        self.left_frame = ttk.LabelFrame(self.h_paned, text="Data Loading Options", padding="5")
        self.h_paned.add(self.left_frame)
        
        # Create right pane with vertical split
        self.v_paned = ttk.PanedWindow(self.h_paned, orient=tk.VERTICAL)
        self.h_paned.add(self.v_paned)
        
        # Create status frame in top right
        self.status_frame = ttk.LabelFrame(self.v_paned, text="Status", padding="5")
        self.v_paned.add(self.status_frame)
        
        # Create data preview frame in bottom right
        self.preview_frame = ttk.LabelFrame(self.v_paned, text="Data Preview", padding="5")
        self.v_paned.add(self.preview_frame)
        
        # Add buttons to left frame
        button_configs = [
            ("Load Eye Tracking Data", self.load_eye_tracking),
            ("Load Face Heatmap Data", self.load_face_heatmap),
            ("Load Vitals Data", self.load_vitals),
            ("Load EEG Data", self.load_eeg),
            ("Load Survey Data", self.load_survey),
            ("Load JSON Data", self.load_json)
        ]
        
        for i, (text, command) in enumerate(button_configs):
            ttk.Button(self.left_frame, text=text, command=command).grid(
                row=i, column=0, pady=5, padx=5, sticky=(tk.W, tk.E)
            )
        
        # Status label in status frame
        self.status_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.status_var).grid(
            row=0, column=0, pady=5, padx=5
        )
        
        # Preview text widget in preview frame
        self.preview_text = tk.Text(self.preview_frame, height=10, width=40)
        self.preview_text.grid(row=0, column=0, pady=5, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar for preview text
        preview_scroll = ttk.Scrollbar(self.preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        preview_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.preview_text.configure(yscrollcommand=preview_scroll.set)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)
        
        self.loaded_data = {}
        
        # Add Analysis and Export frames
        self.right_frame = ttk.Frame(self.h_paned)
        self.h_paned.add(self.right_frame)
        
        # Analysis Options Frame
        self.analysis_frame = ttk.LabelFrame(self.right_frame, text="Analysis Options", padding="5")
        self.analysis_frame.grid(row=0, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        # Analysis buttons
        ttk.Button(self.analysis_frame, text="Preprocess Data", 
                  command=self.preprocess_data).grid(row=0, column=0, pady=5, padx=5)
        ttk.Button(self.analysis_frame, text="Extract Features", 
                  command=self.extract_features).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(self.analysis_frame, text="Analyze Correlations", 
                  command=self.analyze_correlations).grid(row=0, column=2, pady=5, padx=5)
        
        # Export Options Frame
        self.export_frame = ttk.LabelFrame(self.right_frame, text="Export Options", padding="5")
        self.export_frame.grid(row=1, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        # Export buttons
        ttk.Button(self.export_frame, text="Export to CSV", 
                  command=self.export_to_csv).grid(row=0, column=0, pady=5, padx=5)
        ttk.Button(self.export_frame, text="Export to Pickle", 
                  command=self.export_to_pickle).grid(row=0, column=1, pady=5, padx=5)
        
        # Add data processing state
        self.processed_data = {}
        self.features = {}
        self.analysis_results = {}

    def load_eye_tracking(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                data = load_eye_tracking_data(file_path)
                self.loaded_data['eye_tracking'] = data
                self.status_var.set(f"Successfully loaded eye tracking data")
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, str(data.head()))
                
                # Save ingested data
                output_path = INGESTION_DIR / f"eye_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(output_path, index=False)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_face_heatmap(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                data = load_face_heatmap_data(file_path)
                self.loaded_data['face_heatmap'] = data
                self.status_var.set(f"Successfully loaded face heatmap data")
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, str(data.head()))
                
                # Save ingested data
                output_path = INGESTION_DIR / f"face_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(output_path, index=False)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_vitals(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                data = load_vitals_data(file_path)
                self.loaded_data['vitals'] = data
                self.status_var.set(f"Successfully loaded vitals data")
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, str(data.head()))
                
                # Save ingested data
                output_path = INGESTION_DIR / f"vitals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(output_path, index=False)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_eeg(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf")])
            if file_path:
                data = load_eeg_data(file_path)
                self.loaded_data['eeg'] = data
                self.status_var.set(f"Successfully loaded EEG data")
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, str(data.info))
                
                # Save ingested data
                output_path = INGESTION_DIR / f"eeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fif"
                data.save(output_path, overwrite=True)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_survey(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                data = load_survey_data(file_path)
                self.loaded_data['survey'] = data
                self.status_var.set(f"Successfully loaded survey data")
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, str(data.head()))
                
                # Save ingested data
                output_path = INGESTION_DIR / f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(output_path, index=False)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_json(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if file_path:
                data = load_json_data(file_path)
                self.loaded_data['json'] = data
                self.status_var.set(f"Successfully loaded JSON data")
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, str(data))
                
                # Save ingested data
                output_path = INGESTION_DIR / f"json_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def preprocess_data(self):
        """Preprocess the loaded data"""
        try:
            if not self.loaded_data:
                raise ValueError("No data loaded to preprocess")
                
            from src.data_preprocessing import preprocess_multimodal_data
            
            self.processed_data = preprocess_multimodal_data(self.loaded_data)
            self.status_var.set("Data preprocessing completed")
            
            # Show preview of processed data
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "Preprocessed Data Summary:\n\n")
            for key, data in self.processed_data.items():
                self.preview_text.insert(tk.END, f"{key}:\n{str(data.head())}\n\n")
                
            # Save preprocessed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for key, data in self.processed_data.items():
                if isinstance(data, pd.DataFrame):
                    output_path = RESULTS_DIR / f"preprocessed_{key}_{timestamp}.csv"
                    data.to_csv(output_path, index=False)
                
        except Exception as e:
            messagebox.showerror("Preprocessing Error", str(e))
            logger.error(f"Preprocessing error: {str(e)}")

    def extract_features(self):
        """Extract features from preprocessed data"""
        try:
            if not self.processed_data:
                raise ValueError("Please preprocess the data first")
                
            from src.feature_extraction import extract_multimodal_features
            
            self.features = extract_multimodal_features(self.processed_data)
            self.status_var.set("Feature extraction completed")
            
            # Show preview of extracted features
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "Extracted Features Summary:\n\n")
            for key, data in self.features.items():
                self.preview_text.insert(tk.END, f"{key}:\n{str(data.head())}\n\n")
                
            # Save extracted features
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for key, data in self.features.items():
                if isinstance(data, pd.DataFrame):
                    output_path = RESULTS_DIR / f"features_{key}_{timestamp}.csv"
                    data.to_csv(output_path, index=False)
                
        except Exception as e:
            messagebox.showerror("Feature Extraction Error", str(e))
            logger.error(f"Feature extraction error: {str(e)}")

    def analyze_correlations(self):
        """Perform correlation analysis on extracted features"""
        try:
            if not self.features:
                raise ValueError("Please extract features first")
                
            from src.correlation_analysis import analyze_multimodal_correlations
            
            self.analysis_results = analyze_multimodal_correlations(self.features)
            self.status_var.set("Correlation analysis completed")
            
            # Show preview of analysis results
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "Correlation Analysis Results:\n\n")
            for key, result in self.analysis_results.items():
                self.preview_text.insert(tk.END, f"{key}:\n{str(result)}\n\n")
                
            # Save analysis results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for key, result in self.analysis_results.items():
                output_path = RESULTS_DIR / f"correlation_{key}_{timestamp}.csv"
                if isinstance(result, pd.DataFrame):
                    result.to_csv(output_path, index=False)
                else:
                    pd.DataFrame([result]).to_csv(output_path, index=False)
                
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            logger.error(f"Correlation analysis error: {str(e)}")

    def export_to_csv(self):
        """Export processed data and results to CSV files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export each type of data
            data_sets = {
                'processed': (self.processed_data, RESULTS_DIR),
                'features': (self.features, RESULTS_DIR),
                'analysis': (self.analysis_results, RESULTS_DIR)
            }
            
            for data_type, (data_dict, output_dir) in data_sets.items():
                if data_dict:
                    for key, data in data_dict.items():
                        filename = f"{data_type}_{key}_{timestamp}.csv"
                        filepath = output_dir / filename
                        
                        if isinstance(data, pd.DataFrame):
                            data.to_csv(filepath, index=False)
                        elif isinstance(data, dict):
                            pd.DataFrame(data).to_csv(filepath, index=False)
                        else:
                            pd.DataFrame([data]).to_csv(filepath, index=False)
            
            self.status_var.set(f"Data exported to {RESULTS_DIR}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            logger.error(f"CSV export error: {str(e)}")

    def export_to_pickle(self):
        """Export all data to a pickle file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = RESULTS_DIR / f"all_data_{timestamp}.pkl"
            
            export_data = {
                'raw_data': self.loaded_data,
                'processed_data': self.processed_data,
                'features': self.features,
                'analysis_results': self.analysis_results
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f)
                
            self.status_var.set(f"Data exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            logger.error(f"Pickle export error: {str(e)}")

def validate_file_exists(path: Union[str, Path]) -> None:
    """Validate that the file exists at the given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")

def load_eye_tracking_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate eye tracking data from CSV file.
    
    Args:
        path: Path to the eye tracking CSV file
        
    Returns:
        DataFrame containing eye tracking data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        Exception: For other loading errors
    """
    try:
        validate_file_exists(path)
        df = pd.read_csv(path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("Eye tracking data file is empty")
            
        required_columns = ['timestamp', 'x_position', 'y_position']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Successfully loaded eye tracking data from {path} with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load eye tracking data: {str(e)}")
        raise

def load_face_heatmap_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate face heatmap data from CSV file.
    
    Args:
        path: Path to the face heatmap CSV file
        
    Returns:
        DataFrame containing face heatmap data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        Exception: For other loading errors
    """
    try:
        validate_file_exists(path)
        df = pd.read_csv(path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("Face heatmap data file is empty")
            
        required_columns = ['timestamp', 'heatmap_values']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Successfully loaded face heatmap data from {path} with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load face heatmap data: {str(e)}")
        raise

def load_vitals_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate vitals data from CSV file.
    
    Args:
        path: Path to the vitals CSV file
        
    Returns:
        DataFrame containing vitals data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        Exception: For other loading errors
    """
    try:
        validate_file_exists(path)
        df = pd.read_csv(path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("Vitals data file is empty")
            
        required_columns = ['timestamp', 'heart_rate', 'blood_pressure', 'temperature']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Successfully loaded vitals data from {path} with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load vitals data: {str(e)}")
        raise

def load_eeg_data(path: Union[str, Path]) -> mne.io.Raw:
    """
    Load and validate EEG data from EDF file using MNE.
    
    Args:
        path: Path to the EEG EDF file
        
    Returns:
        MNE Raw object containing EEG data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other loading errors
    """
    try:
        validate_file_exists(path)
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        
        if len(raw.ch_names) == 0:
            raise ValueError("EEG data file contains no channels")
            
        logger.info(f"Successfully loaded EEG data from {path} with {len(raw.ch_names)} channels")
        return raw
        
    except Exception as e:
        logger.error(f"Failed to load EEG data: {str(e)}")
        raise

def load_survey_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate survey data from CSV file.
    
    Args:
        path: Path to the survey CSV file
        
    Returns:
        DataFrame containing survey data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        Exception: For other loading errors
    """
    try:
        validate_file_exists(path)
        df = pd.read_csv(path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("Survey data file is empty")
            
        required_columns = ['participant_id', 'question', 'response']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Successfully loaded survey data from {path} with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load survey data: {str(e)}")
        raise

def load_json_data(path: Union[str, Path]) -> Dict:
    """
    Load and validate JSON data from file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary containing JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If invalid JSON
        Exception: For other loading errors
    """
    try:
        validate_file_exists(path)
        with open(path, 'r') as file:
            data = json.load(file)
            
        if not data:
            raise ValueError("JSON data file is empty")
            
        logger.info(f"Successfully loaded JSON data from {path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON data: {str(e)}")
        raise

def get_data_files(data_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Recursively find all data files in a directory.
    
    Args:
        data_dir: Directory path to search for data files
        
    Returns:
        Dictionary mapping file names (without extension) to full file paths
        
    Raises:
        NotADirectoryError: If data_dir doesn't exist or isn't a directory
    """
    try:
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Directory not found: {data_dir}")
            
        data_files = {}
        valid_extensions = ('.csv', '.edf', '.json')
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(valid_extensions):
                    key = os.path.splitext(file)[0]
                    data_files[key] = os.path.join(root, file)
                    
        if not data_files:
            logger.warning(f"No data files found in {data_dir}")
        else:
            logger.info(f"Found {len(data_files)} data files in {data_dir}")
            
        return data_files
        
    except Exception as e:
        logger.error(f"Error scanning directory for data files: {str(e)}")
        raise

if __name__ == "__main__":
    root = tk.Tk()
    app = DataIngestionGUI(root)
    root.mainloop()
