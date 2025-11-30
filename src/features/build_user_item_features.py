"""
Feature Engineering Module for User-Item Interaction Features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from ..utils.config import FeatureConfig
from ..utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

class UserItemFeatureBuilder:
    """Build user-item interaction features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    def build_features(self,
                      orders: pd.DataFrame,
                      order_products: pd.DataFrame,
                      user_features: pd.DataFrame,
                      item_features: pd.DataFrame) -> pd.DataFrame:
        """
        Build user-item interaction features
        
        Args:
            orders: Orders dataframe
            order_products: Order products dataframe
            user_features: User features dataframe
            item_features: Item features dataframe
            
        Returns:
            DataFrame with user-item features
        """
        logger.info("Building user-item interaction features...")
        
        # Create base user-item pairs from historical interactions
        user_item_pairs = self._create_user_item_pairs(orders, order_products)
        
        # Build interaction features
        interaction_features = self._build_interaction_features(
            orders, order_products, user_item_pairs
        )
        
        # Add user features
        interaction_features = interaction_features.merge(
            user_features, on='user_id', how='left'
        )
        
        # Add item features
        interaction_features = interaction_features.merge(
            item_features, on='product_id', how='left'
        )
        
        # Build cross features
        cross_features = self._build_cross_features(interaction_features)
        interaction_features = pd.concat([interaction_features, cross_features], axis=1)
        
        # Fill missing values
        numeric_columns = interaction_features.select_dtypes(include=[np.number]).columns
        interaction_features[numeric_columns] = interaction_features[numeric_columns].fillna(0)
        
        logger.info(f"Built user-item features with {len(interaction_features)} interactions and {len(interaction_features.columns)} features")
        
        return interaction_features
    
    def _create_user_item_pairs(self, 
                               orders: pd.DataFrame,
                               order_products: pd.DataFrame) -> pd.DataFrame:
        """Create user-item pairs from historical data"""
        
        # Get user-product interactions
        user_item_pairs = order_products.merge(
            orders[['order_id', 'user_id']], on='order_id'
        )[['user_id', 'product_id']].drop_duplicates()
        
        return user_item_pairs
    
    def _build_interaction_features(self,
                                   orders: pd.DataFrame,
                                   order_products: pd.DataFrame,
                                   user_item_pairs: pd.DataFrame) -> pd.DataFrame:
        """Build user-item interaction features"""
        
        # Merge orders with order_products
        user_orders = order_products.merge(
            orders[['order_id', 'user_id', 'order_number', 'days_since_prior_order']], 
            on='order_id'
        )
        
        # Group by user-item pairs for aggregation
        interaction_agg = user_orders.groupby(['user_id', 'product_id']).agg({
            'order_id': 'nunique',  # Number of times ordered
            'reordered': 'sum',     # Number of reorders
            'add_to_cart_order': ['mean', 'std', 'min', 'max'],  # Cart position stats
            'order_number': ['min', 'max', 'mean'],  # Order timing
            'days_since_prior_order': 'mean'  # Average days between orders
        }).round(4)
        
        # Flatten column names
        interaction_agg.columns = [
            'user_item_orders', 'user_item_reorders', 
            'avg_cart_position', 'std_cart_position', 'min_cart_position', 'max_cart_position',
            'first_order_number', 'last_order_number', 'avg_order_number',
            'avg_days_between_orders'
        ]
        
        # Additional derived features
        interaction_agg['user_item_reorder_rate'] = (
            interaction_agg['user_item_reorders'] / interaction_agg['user_item_orders']
        ).fillna(0)
        
        # Order span (how many orders between first and last purchase)
        interaction_agg['order_span'] = (
            interaction_agg['last_order_number'] - interaction_agg['first_order_number'] + 1
        )
        
        # Purchase frequency (orders per order span)
        interaction_agg['purchase_frequency'] = (
            interaction_agg['user_item_orders'] / interaction_agg['order_span']
        ).fillna(0)
        
        # Recency features
        user_max_orders = orders.groupby('user_id')['order_number'].max().reset_index()
        user_max_orders.columns = ['user_id', 'user_max_order']
        
        interaction_features = interaction_agg.reset_index()
        interaction_features = interaction_features.merge(user_max_orders, on='user_id')
        
        # Days since last purchase (relative to user's last order)
        interaction_features['orders_since_last_purchase'] = (
            interaction_features['user_max_order'] - interaction_features['last_order_number']
        )
        
        # Normalize by user's total orders
        user_order_counts = orders.groupby('user_id')['order_id'].nunique().reset_index()
        user_order_counts.columns = ['user_id', 'user_total_orders']
        
        interaction_features = interaction_features.merge(user_order_counts, on='user_id')
        interaction_features['user_item_order_ratio'] = (
            interaction_features['user_item_orders'] / interaction_features['user_total_orders']
        )
        
        interaction_features = interaction_features.fillna(0)
        
        return interaction_features
    
    def _build_cross_features(self, interaction_features: pd.DataFrame) -> pd.DataFrame:
        """Build cross features between user and item characteristics"""
        
        cross_features = pd.DataFrame(index=interaction_features.index)
        
        # User basket size vs item popularity
        if 'avg_basket_size' in interaction_features.columns and 'total_orders' in interaction_features.columns:
            cross_features['basket_size_item_popularity'] = (
                interaction_features['avg_basket_size'] * interaction_features['total_orders']
            ).round(4)
        
        # User reorder rate vs item reorder rate
        if 'reorder_rate' in interaction_features.columns and 'reorder_rate' in interaction_features.columns:
            cross_features['user_item_reorder_alignment'] = (
                interaction_features['reorder_rate'] * interaction_features.get('reorder_rate', 0)
            ).round(4)
        
        # User order frequency vs item purchase frequency
        if 'avg_days_since_prior' in interaction_features.columns:
            user_order_freq = 1 / (interaction_features['avg_days_since_prior'] + 1)  # Avoid division by zero
            cross_features['user_order_frequency'] = user_order_freq.round(4)
            
            if 'purchase_frequency' in interaction_features.columns:
                cross_features['user_item_frequency_match'] = (
                    user_order_freq * interaction_features['purchase_frequency']
                ).round(4)
        
        # User diversity vs item ubiquity
        if 'unique_products' in interaction_features.columns and 'total_users' in interaction_features.columns:
            cross_features['diversity_ubiquity'] = (
                interaction_features['unique_products'] / (interaction_features['total_users'] + 1)
            ).round(4)
        
        # Department/aisle affinity
        if 'unique_departments' in interaction_features.columns and 'department_encoded' in interaction_features.columns:
            cross_features['dept_exploration_vs_item_dept'] = (
                interaction_features['unique_departments'] * interaction_features['department_encoded']
            ).round(4)
        
        return cross_features