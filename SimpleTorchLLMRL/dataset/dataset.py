import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Union, List, Dict
from datasets import Dataset as HFDataset, load_dataset
import pandas as pd


class Dataset(TorchDataset):
    
    def __init__(self, data: List[Dict[str, str]]):
        """
        PyTorch Dataset for question-answer pairs.
        
        Args:
            data: List of dictionaries with "question" and "solution" keys
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        return self.data[idx]
    
    def get_batch(self, indices):
        """Get multiple items at once."""
        batch_data = [self.data[i] for i in indices]
        return {
            "questions": [item["question"] for item in batch_data],
            "solutions": [item["solution"] for item in batch_data]
        }
    
    @classmethod
    def from_huggingface(cls, 
                        dataset_name: str, 
                        question_col: str = "question", 
                        solution_col: str = "solution",
                        split: str = "train"):
        """
        Create dataset from Hugging Face dataset.
        
        Args:
            dataset_name: Name of the HF dataset
            question_col: Column name for questions
            solution_col: Column name for solutions
            split: Dataset split to use
        """
        hf_dataset = load_dataset(dataset_name, split=split)
        
        # Validate columns
        if question_col not in hf_dataset.column_names:
            raise ValueError(f"Question column '{question_col}' not found. Available: {hf_dataset.column_names}")
        if solution_col not in hf_dataset.column_names:
            raise ValueError(f"Solution column '{solution_col}' not found. Available: {hf_dataset.column_names}")
        
        # Convert to our format
        data = []
        for item in hf_dataset:
            data.append({
                "question": item[question_col],
                "solution": item[solution_col]
            })
        
        return cls(data)
    
    @classmethod
    def from_csv(cls, 
                 file_path: str, 
                 question_col: str = "question", 
                 solution_col: str = "solution"):
        """
        Create dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            question_col: Column name for questions
            solution_col: Column name for solutions
        """
        df = pd.read_csv(file_path)
        
        # Validate columns
        if question_col not in df.columns:
            raise ValueError(f"Question column '{question_col}' not found. Available: {list(df.columns)}")
        if solution_col not in df.columns:
            raise ValueError(f"Solution column '{solution_col}' not found. Available: {list(df.columns)}")
        
        # Convert to our format
        data = []
        for _, row in df.iterrows():
            data.append({
                "question": str(row[question_col]),
                "solution": str(row[solution_col])
            })
        
        return cls(data)
    
    @classmethod
    def from_txt(cls, 
                 file_path: str, 
                 separator: str = "\t"):
        """
        Create dataset from text file.
        
        Args:
            file_path: Path to text file
            separator: Separator between question and solution (default: tab)
        
        Expected format: question<separator>solution per line
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(separator)
                if len(parts) != 2:
                    raise ValueError(f"Line {line_num}: Expected 2 parts separated by '{separator}', got {len(parts)}")
                
                question, solution = parts
                data.append({
                    "question": question.strip(),
                    "solution": solution.strip()
                })
        
        return cls(data)
    
    @classmethod
    def from_json(cls, 
                  file_path: str, 
                  question_key: str = "question", 
                  solution_key: str = "solution"):
        """
        Create dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            question_key: Key name for questions in JSON
            solution_key: Key name for solutions in JSON
        
        Expected format: List of objects with question_key and solution_key
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, list):
            raise ValueError("JSON file must contain a list of objects")
        
        data = []
        for i, item in enumerate(json_data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i}: Expected dict, got {type(item)}")
            
            if question_key not in item:
                raise ValueError(f"Item {i}: Missing key '{question_key}'")
            if solution_key not in item:
                raise ValueError(f"Item {i}: Missing key '{solution_key}'")
            
            data.append({
                "question": str(item[question_key]),
                "solution": str(item[solution_key])
            })
        
        return cls(data)
    