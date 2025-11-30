"""
Sequence Builder for Sequential Models (LSTM, TCN)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.preprocessing import LabelEncoder
from ..utils.config import FeatureConfig
from ..utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

class SequenceBuilder:
    """Build sequences for sequential deep learning models"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.product_encoder = LabelEncoder()
        self.vocab_size = None
        
    def build_sequences(self,
                       orders: pd.DataFrame,
                       order_products: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Build product sequences for each user
        
        Args:
            orders: Orders dataframe
            order_products: Order products dataframe
            
        Returns:
            Tuple of (sequences, targets, metadata)
        """
        logger.info("Building user product sequences...")
        
        # Merge orders with products
        user_orders = self._prepare_user_orders(orders, order_products)
        
        # Fit product encoder on most frequent products
        self._fit_product_encoder(user_orders)
        
        # Build sequences per user
        sequences, targets, user_ids = self._build_user_sequences(user_orders)
        
        # Pad sequences
        sequences_padded = self._pad_sequences(sequences)
        
        metadata = {
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.config.max_sequence_length,
            'total_sequences': len(sequences_padded),
            'total_targets': len(targets),
            'product_encoder': self.product_encoder
        }
        
        logger.info(f"Built {len(sequences_padded)} sequences with vocab size {self.vocab_size}")
        
        return sequences_padded, np.array(targets), metadata
    
    def _prepare_user_orders(self, 
                            orders: pd.DataFrame,
                            order_products: pd.DataFrame) -> pd.DataFrame:
        """Prepare user orders sorted by order number"""
        
        # Merge orders with products
        user_orders = orders.merge(order_products, on='order_id')
        
        # Sort by user and order number
        user_orders = user_orders.sort_values(['user_id', 'order_number', 'add_to_cart_order'])
        
        return user_orders
    
    def _fit_product_encoder(self, user_orders: pd.DataFrame):
        """Fit product encoder on most frequent products"""
        
        # Get product frequencies
        product_counts = user_orders['product_id'].value_counts()
        
        # Limit vocabulary size if specified
        if self.config.vocab_size_limit:
            top_products = product_counts.head(self.config.vocab_size_limit).index
            # Add special token for unknown products
            vocab_products = ['<UNK>'] + top_products.tolist()
        else:
            vocab_products = ['<UNK>'] + product_counts.index.tolist()
        
        # Fit encoder
        self.product_encoder.fit(vocab_products)
        self.vocab_size = len(vocab_products)
        
        logger.info(f"Product vocabulary size: {self.vocab_size}")
    
    def _encode_products(self, products: pd.Series) -> np.ndarray:
        """Encode products, handling unknown products"""
        
        # Convert unknown products to '<UNK>'
        known_products = self.product_encoder.classes_[1:]  # Exclude '<UNK>'
        products_encoded = products.apply(lambda x: x if x in known_products else '<UNK>')
        
        return self.product_encoder.transform(products_encoded)
    
    def _build_user_sequences(self, user_orders: pd.DataFrame) -> Tuple[List, List, List]:
        """Build sequences for each user"""
        
        sequences = []
        targets = []  
        user_ids = []
        
        for user_id, user_data in user_orders.groupby('user_id'):
            # Get user's product sequence
            user_products = user_data['product_id'].tolist()
            
            # Skip users with too few products
            if len(user_products) < 2:
                continue
            
            # Encode products
            encoded_products = self._encode_products(pd.Series(user_products))
            
            # Create sequences with sliding window
            user_sequences, user_targets = self._create_sliding_windows(encoded_products)
            
            sequences.extend(user_sequences)
            targets.extend(user_targets)
            user_ids.extend([user_id] * len(user_sequences))
        
        return sequences, targets, user_ids
    
    def _create_sliding_windows(self, encoded_products: np.ndarray) -> Tuple[List, List]:
        """Create sliding windows from product sequence"""
        
        sequences = []
        targets = []
        
        # Create sequences of increasing length
        for i in range(1, len(encoded_products)):
            # Input sequence (all products up to position i)
            sequence = encoded_products[:i]
            
            # Target (next product)
            target = encoded_products[i]
            
            sequences.append(sequence.tolist())
            targets.append(target)
        
        return sequences, targets
    
    def _pad_sequences(self, sequences: List[List]) -> np.ndarray:
        """Pad sequences to fixed length"""
        
        max_len = self.config.max_sequence_length
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) >= max_len:
                # Truncate to max length (take the most recent items)
                padded_seq = seq[-max_len:]
            else:
                # Pad with zeros (0 is reserved for padding)
                padded_seq = [0] * (max_len - len(seq)) + seq
            
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences)
    
    def build_inference_sequences(self, 
                                 user_orders: pd.DataFrame,
                                 user_ids: List[int]) -> np.ndarray:
        """Build sequences for inference (last N products per user)"""
        
        inference_sequences = []
        
        for user_id in user_ids:
            user_data = user_orders[user_orders['user_id'] == user_id]
            
            if len(user_data) == 0:
                # Empty sequence for new users
                sequence = [0] * self.config.max_sequence_length
            else:
                # Get user's recent products
                user_products = user_data.sort_values(['order_number', 'add_to_cart_order'])['product_id'].tolist()
                encoded_products = self._encode_products(pd.Series(user_products))
                
                # Take last max_sequence_length products
                if len(encoded_products) >= self.config.max_sequence_length:
                    sequence = encoded_products[-self.config.max_sequence_length:].tolist()
                else:
                    # Pad with zeros
                    padding = [0] * (self.config.max_sequence_length - len(encoded_products))
                    sequence = padding + encoded_products.tolist()
            
            inference_sequences.append(sequence)
        
        return np.array(inference_sequences)
    
    def decode_products(self, encoded_products: np.ndarray) -> List:
        """Decode products back to original IDs"""
        
        # Filter out padding tokens (0)
        valid_encoded = encoded_products[encoded_products > 0]
        
        if len(valid_encoded) == 0:
            return []
        
        # Decode products
        decoded = self.product_encoder.inverse_transform(valid_encoded)
        
        # Filter out unknown tokens
        decoded_products = [p for p in decoded if p != '<UNK>']
        
        return decoded_products