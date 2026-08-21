from deepmol.compound_featurization import MolecularFeaturizer
from rdkit.Chem import Mol
import numpy as np
import logging
from typing import List, Optional, Union
import warnings
import os
from functools import lru_cache

# Force SafeTensors for compatibility
os.environ["TRANSFORMERS_USE_SAFETENSORS"] = "true"

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _transformers_available = True
except ImportError:
    _transformers_available = False

# Global model cache to avoid reloading models
_MODEL_CACHE = {}

def _get_model_cache_key(model_name: str, pooling: str) -> str:
    """Generate cache key for model instances."""
    return f"{model_name}_{pooling}"

class HuggingFaceFeaturizer(MolecularFeaturizer):
    """
    HuggingFace featurizer that generates molecular embeddings using pre-trained
    Hugging Face models.
    
    HuggingFace models are transformer models pre-trained on large-scale molecular 
    datasets using SMILES strings, adapting NLP techniques to chemistry.
    
    Features:
    - Batch processing 
    - Model caching to avoid reloading
    - Progress tracking for large datasets
    - Multiple ChemBERTa variant support
    - GPU/CPU auto-detection
    
    Parameters
    ----------
    model_name : str, default "seyonec/ChemBERTa-zinc-base-v1"
        Name of the pre-trained ChemBERTa model from Hugging Face Hub.
        Supported models:
        - "seyonec/ChemBERTa-zinc-base-v1" (default, 768 dim)
        - "seyonec/PubChem10M_SMILES_BPE_396_250" (768 dim)
        - "DeepChem/ChemBERTa-77M-MLM" (384 dim)
        - "DeepChem/ChemBERTa-77M-MTR" (384 dim)
    pooling : str, default "mean"
        Pooling strategy to generate molecule-level embeddings from token embeddings.
        Options: "mean", "cls"
    max_length : int, default 512
        Maximum sequence length for tokenization.
    device : str, optional
        Device to run the model on ('cuda', 'cpu', or None for auto-detection)
    batch_size : int, default 32
        Batch size for processing multiple molecules (only affects batch_featurize method)
    cache_models : bool, default True
        Whether to cache models to avoid reloading
    """
    
    def __init__(self, 
                 model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 pooling: str = "mean",
                 max_length: int = 512,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 cache_models: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        if not _transformers_available:
            raise ImportError(
                "HuggingFaceFeaturizer requires transformers and torch to be installed. "
                "Please install them using: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_models = cache_models
        
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._initialize_model()
        
        # Feature names will be set after model initialization
        self.feature_names = None
        self._initialize_feature_names()
        
        logger.info(f"Initialized HuggingFaceFeaturizer with model: {model_name}, "
                   f"device: {self.device}, pooling: {pooling}")
    
    def _initialize_model(self):
        """Initialize the tokenizer and model with caching support."""
        cache_key = _get_model_cache_key(self.model_name, self.pooling) if self.cache_models else None
        
        # Check cache first
        if self.cache_models and cache_key in _MODEL_CACHE:
            logger.info(f"Loading cached model: {cache_key}")
            self.tokenizer, self.model = _MODEL_CACHE[cache_key]
            self.model = self.model.to(self.device)
            return
        
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Check if tokenizer has a pad token, if not set it
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
            
            # Load model with SafeTensors and optimized settings
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Cache the model if requested
            if self.cache_models and cache_key:
                # Store a copy on CPU for caching to avoid GPU memory issues
                cached_model = self.model.cpu()
                _MODEL_CACHE[cache_key] = (self.tokenizer, cached_model)
                self.model = self.model.to(self.device)
            
            logger.info(f"Successfully loaded ChemBERTa model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load ChemBERTa model {self.model_name}: {str(e)}")
            logger.info("Attempting to load alternative models...")
            self._try_alternative_models()
    
    def _try_alternative_models(self):
        """Try loading alternative models if primary fails."""
        alternative_models = [
            "seyonec/ChemBERTa-zinc-base-v1",
            "seyonec/PubChem10M_SMILES_BPE_396_250", 
            "DeepChem/ChemBERTa-77M-MLM",
            "DeepChem/ChemBERTa-77M-MTR"
        ]
        
        # Remove the current model name if it's in the list
        alternative_models = [m for m in alternative_models if m != self.model_name]
        
        for model_name in alternative_models:
            try:
                logger.info(f"Trying alternative model: {model_name}")
                self.model_name = model_name
                self._initialize_model()
                logger.info(f"Successfully loaded alternative model: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        raise ImportError("Could not load any ChemBERTa model. Please check your internet connection and model names.")
    
    def _initialize_feature_names(self):
        """Initialize feature names based on model embedding dimension."""
        if self.model is not None:
            # Get embedding dimension from the model config
            embedding_dim = self.model.config.hidden_size
            self.feature_names = [f'chemberta_{i}' for i in range(embedding_dim)]
            logger.debug(f"Initialized {embedding_dim} feature names")
        else:
            # Fallback dimensions based on common models
            fallback_dims = {
                "seyonec/PubChem10M_SMILES_BPE_396_250": 256,
                "default": 768
            }
            dim = fallback_dims.get(self.model_name, fallback_dims["default"])
            self.feature_names = [f'chemberta_{i}' for i in range(dim)]
            warnings.warn(f"Model not initialized, using default feature dimension of {dim}")
    
    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Featurize a single RDKit molecule object.
    
        Parameters
        ----------
        mol : Mol
            RDKit molecule object

        Returns
        -------
        np.ndarray
            Molecular embedding vector
        
        Raises
        ------
        ValueError
            If the molecule is invalid or cannot be converted to SMILES
        """
        from rdkit.Chem import MolToSmiles
    
        # Check for invalid molecule
        if mol is None:
            raise ValueError("Invalid molecule: None")
        
        # Convert RDKit Mol to SMILES
        try:
            smiles = MolToSmiles(mol, isomericSmiles=False)
            if not smiles:
                raise ValueError("Invalid molecule or unable to convert to SMILES")
        except Exception as e:
            raise ValueError(f"Failed to convert molecule to SMILES: {str(e)}")
        
        return self._featurize_smiles(smiles)
    
    def _featurize_smiles(self, smiles: str) -> np.ndarray:
        """
        Featurize a single SMILES string.
        
        Parameters
        ----------
        smiles : str
            SMILES string
            
        Returns
        -------
        np.ndarray
            Molecular embedding vector
            
        Raises
        ------
        ValueError
            If the SMILES string is invalid
        """
        if not smiles or not isinstance(smiles, str):
            raise ValueError("Invalid SMILES string")
                
        # Tokenize SMILES string
        inputs = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
        
        # Apply pooling strategy to get molecule-level embedding
        if self.pooling == "mean":
            # Mean pooling excluding padding tokens
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        elif self.pooling == "cls":
            # Use [CLS] token embedding
            embedding = token_embeddings[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
        
        # Convert to numpy and ensure 1D array
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Verify the embedding has the expected dimension
        expected_dim = len(self.feature_names)
        if embedding_np.shape[0] != expected_dim:
            raise ValueError(f"Embedding dimension {embedding_np.shape[0]} doesn't match expected {expected_dim}")
        
        return embedding_np.astype(np.float32)
    
    def batch_featurize(self, molecules: List[Mol], show_progress: bool = False) -> np.ndarray:
        """
        Featurize a batch of molecules for better performance.
        
        Parameters
        ----------
        molecules : List[Mol]
            List of RDKit molecule objects
        show_progress : bool, default False
            Whether to show progress bar
            
        Returns
        -------
        np.ndarray
            2D array of molecular embeddings
        """
        from rdkit.Chem import MolToSmiles
        from tqdm import tqdm
        
        valid_smiles = []
        valid_indices = []
        
        # Convert molecules to SMILES and filter invalid ones
        for i, mol in enumerate(molecules):
            if mol is not None:
                smiles = MolToSmiles(mol, isomericSmiles=False)
                if smiles:
                    valid_smiles.append(smiles)
                    valid_indices.append(i)
        
        if not valid_smiles:
            logger.warning("No valid molecules found in batch")
            return np.full((len(molecules), len(self.feature_names)), np.nan, dtype=np.float32)
        
        # Process in batches
        embeddings = []
        iterator = range(0, len(valid_smiles), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Featurizing molecules")
        
        for start_idx in iterator:
            end_idx = min(start_idx + self.batch_size, len(valid_smiles))
            batch_smiles = valid_smiles[start_idx:end_idx]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # Generate embeddings for batch
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    token_embeddings = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling == "mean":
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                else:  # cls
                    batch_embeddings = token_embeddings[:, 0, :]
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Failed to process batch {start_idx}-{end_idx}: {str(e)}")
                # Add NaN embeddings for failed batch
                batch_size = len(batch_smiles)
                failed_embeddings = np.full((batch_size, len(self.feature_names)), np.nan, dtype=np.float32)
                embeddings.append(failed_embeddings)
        
        # Combine all batches
        if embeddings:
            all_embeddings = np.vstack(embeddings)
        else:
            all_embeddings = np.array([])
        
        # Create full results array with NaN for invalid molecules
        full_embeddings = np.full((len(molecules), len(self.feature_names)), np.nan, dtype=np.float32)
        if all_embeddings.size > 0:
            full_embeddings[valid_indices] = all_embeddings
        
        return full_embeddings
    
    
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return len(self.feature_names)
    
    def clear_model_cache():
        """Clear the global model cache."""
        global _MODEL_CACHE
        _MODEL_CACHE.clear()
        logger.info("Cleared ChemBERTa model cache")
    
    @property
    def supported_models(self) -> List[str]:
        """Get list of supported ChemBERTa models."""
        return [
            "seyonec/ChemBERTa-zinc-base-v1",
            "seyonec/PubChem10M_SMILES_BPE_396_250",
            "DeepChem/ChemBERTa-77M-MLM", 
            "DeepChem/ChemBERTa-77M-MTR"
        ]


