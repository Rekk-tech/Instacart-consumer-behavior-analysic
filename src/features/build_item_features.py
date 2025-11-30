"""
Feature Engineering Module for Item Features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from ..utils.config import FeatureConfig
from ..utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

class ItemFeatureBuilder:
    """Build comprehensive item/product features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    def build_features(self,
                      orders: pd.DataFrame,
                      order_products: pd.DataFrame, 
                      products: pd.DataFrame,
                      aisles: pd.DataFrame,
                      departments: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive item features
        
        Args:
            orders: Orders dataframe
            order_products: Order products dataframe
            products: Products dataframe  
            aisles: Aisles dataframe
            departments: Departments dataframe
            
        Returns:
            DataFrame with item features
        """
        logger.info("Building item features...")
        
        # Merge with orders for temporal info
        op_with_orders = order_products.merge(orders[['order_id', 'user_id', 'order_number']], on='order_id')
        
        # Base product info
        item_features = products.copy()
        
        feature_groups = []
        
        # Build different feature categories
        if 'popularity' in self.config.item_feature_categories:
            popularity_features = self._build_popularity_features(op_with_orders)
            feature_groups.append(popularity_features)
            
        if 'reorder_patterns' in self.config.item_feature_categories:
            reorder_features = self._build_reorder_pattern_features(op_with_orders)
            feature_groups.append(reorder_features)
            
        if 'department_aisle' in self.config.item_feature_categories:
            dept_aisle_features = self._build_department_aisle_features(products, aisles, departments)
            feature_groups.append(dept_aisle_features)
            
        # Merge all feature groups with base product info
        for features in feature_groups:
            item_features = item_features.merge(features, on='product_id', how='left')
            
        # Fill missing values
        numeric_columns = item_features.select_dtypes(include=[np.number]).columns
        item_features[numeric_columns] = item_features[numeric_columns].fillna(0)
        
        logger.info(f"Built {len(item_features.columns)-1} item features for {len(item_features)} products")
        
        return item_features
    
    def _build_popularity_features(self, op_with_orders: pd.DataFrame) -> pd.DataFrame:
        """Build product popularity features"""
        
        popularity_features = op_with_orders.groupby('product_id').agg({
            'order_id': 'nunique',  # Number of unique orders
            'user_id': 'nunique',   # Number of unique users
            'add_to_cart_order': ['mean', 'std']  # Cart position stats
        }).round(4)
        
        # Flatten column names
        popularity_features.columns = [
            'total_orders', 'total_users', 'avg_cart_position', 'std_cart_position'
        ]
        
        # Additional popularity metrics
        popularity_features['orders_per_user'] = (
            popularity_features['total_orders'] / popularity_features['total_users']
        ).round(4)
        
        # Global popularity percentiles
        popularity_features['popularity_rank'] = popularity_features['total_orders'].rank(pct=True)
        popularity_features['user_reach_rank'] = popularity_features['total_users'].rank(pct=True)
        
        popularity_features = popularity_features.fillna(0)
        
        return popularity_features.reset_index()
    
    def _build_reorder_pattern_features(self, op_with_orders: pd.DataFrame) -> pd.DataFrame:
        """Build reorder pattern features"""
        
        # Basic reorder statistics
        reorder_features = op_with_orders.groupby('product_id').agg({
            'reordered': ['mean', 'sum', 'count']
        }).round(4)
        
        # Flatten column names
        reorder_features.columns = ['reorder_rate', 'total_reorders', 'total_purchases']
        
        # Advanced reorder patterns
        # Calculate reorder probability by order number (customer lifecycle stage)
        order_reorder = op_with_orders.groupby(['product_id', 'order_number'])['reordered'].mean().reset_index()
        
        # Early vs late reorder rates (first 5 orders vs later orders)
        early_reorders = order_reorder[order_reorder['order_number'] <= 5].groupby('product_id')['reordered'].mean()
        late_reorders = order_reorder[order_reorder['order_number'] > 5].groupby('product_id')['reordered'].mean()
        
        reorder_features['early_reorder_rate'] = early_reorders
        reorder_features['late_reorder_rate'] = late_reorders
        
        # Reorder consistency (std of reorder rates across users)
        user_product_reorders = op_with_orders.groupby(['user_id', 'product_id'])['reordered'].mean().reset_index()
        reorder_consistency = user_product_reorders.groupby('product_id')['reordered'].std()
        reorder_features['reorder_consistency'] = reorder_consistency
        
        # First purchase vs reorder ratio
        first_purchases = op_with_orders[op_with_orders['reordered'] == 0].groupby('product_id').size()
        reorder_features['first_purchase_count'] = first_purchases
        reorder_features['reorder_to_first_ratio'] = (
            reorder_features['total_reorders'] / reorder_features['first_purchase_count']
        ).fillna(0)
        
        reorder_features = reorder_features.fillna(0)
        
        return reorder_features.reset_index()
    
    def _build_department_aisle_features(self, 
                                       products: pd.DataFrame,
                                       aisles: pd.DataFrame, 
                                       departments: pd.DataFrame) -> pd.DataFrame:
        """Build department and aisle features"""
        
        # Start with products
        dept_aisle_features = products[['product_id', 'aisle_id', 'department_id']].copy()
        
        # Merge aisle info
        dept_aisle_features = dept_aisle_features.merge(
            aisles[['aisle_id', 'aisle']], on='aisle_id', how='left'
        )
        
        # Merge department info
        dept_aisle_features = dept_aisle_features.merge(
            departments[['department_id', 'department']], on='department_id', how='left'
        )
        
        # Create categorical encodings (can be used for embeddings)
        dept_aisle_features['aisle_encoded'] = dept_aisle_features['aisle_id'].astype('category').cat.codes
        dept_aisle_features['department_encoded'] = dept_aisle_features['department_id'].astype('category').cat.codes
        
        # Keep only necessary columns
        keep_cols = ['product_id', 'aisle_id', 'department_id', 'aisle', 'department', 
                     'aisle_encoded', 'department_encoded']
        dept_aisle_features = dept_aisle_features[keep_cols]
        
        return dept_aisle_features