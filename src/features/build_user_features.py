"""
Feature Engineering Module for User Features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from ..utils.config import FeatureConfig
from ..utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

class UserFeatureBuilder:
    """Build comprehensive user features from order history"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    def build_features(self, 
                      orders: pd.DataFrame,
                      order_products: pd.DataFrame,
                      products: pd.DataFrame,
                      aisles: pd.DataFrame,
                      departments: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive user features
        
        Args:
            orders: Orders dataframe
            order_products: Order products dataframe  
            products: Products dataframe
            aisles: Aisles dataframe
            departments: Departments dataframe
            
        Returns:
            DataFrame with user features
        """
        logger.info("Building user features...")
        
        # Merge product information
        op_enriched = self._enrich_order_products(
            order_products, products, aisles, departments
        )
        
        # Merge with orders
        user_orders = orders.merge(op_enriched, on='order_id')
        
        feature_groups = []
        
        # Build different feature categories
        if 'order_patterns' in self.config.user_feature_categories:
            order_features = self._build_order_pattern_features(orders)
            feature_groups.append(order_features)
            
        if 'basket_patterns' in self.config.user_feature_categories:
            basket_features = self._build_basket_pattern_features(user_orders)
            feature_groups.append(basket_features)
            
        if 'product_patterns' in self.config.user_feature_categories:
            product_features = self._build_product_pattern_features(user_orders)
            feature_groups.append(product_features)
            
        if 'department_aisle' in self.config.user_feature_categories:
            dept_aisle_features = self._build_department_aisle_features(user_orders)
            feature_groups.append(dept_aisle_features)
            
        # Merge all feature groups
        user_features = feature_groups[0]
        for features in feature_groups[1:]:
            user_features = user_features.merge(features, on='user_id', how='outer')
            
        logger.info(f"Built {len(user_features.columns)-1} user features for {len(user_features)} users")
        
        return user_features
    
    def _enrich_order_products(self, 
                              order_products: pd.DataFrame,
                              products: pd.DataFrame, 
                              aisles: pd.DataFrame,
                              departments: pd.DataFrame) -> pd.DataFrame:
        """Enrich order products with product metadata"""
        
        # Merge products
        enriched = order_products.merge(products, on='product_id', how='left')
        
        # Merge aisles
        enriched = enriched.merge(aisles, on='aisle_id', how='left')
        
        # Merge departments
        enriched = enriched.merge(departments, on='department_id', how='left')
        
        return enriched
    
    def _build_order_pattern_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Build order pattern features"""
        
        # Order timing features
        orders['order_dow'] = orders['order_dow'].astype(int)
        orders['order_hour_of_day'] = orders['order_hour_of_day'].astype(int)
        
        user_order_features = orders.groupby('user_id').agg({
            'order_number': ['count', 'max', 'mean'],
            'days_since_prior_order': ['mean', 'std', 'min', 'max'],
            'order_dow': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
            'order_hour_of_day': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean()
        }).round(4)
        
        # Flatten column names
        user_order_features.columns = [
            'total_orders', 'max_order_number', 'avg_order_number',
            'avg_days_since_prior', 'std_days_since_prior', 'min_days_since_prior', 'max_days_since_prior',
            'preferred_dow', 'preferred_hour'
        ]
        
        # Fill NaN values
        user_order_features = user_order_features.fillna(0)
        
        return user_order_features.reset_index()
    
    def _build_basket_pattern_features(self, user_orders: pd.DataFrame) -> pd.DataFrame:
        """Build basket pattern features"""
        
        # Calculate basket size per order
        basket_sizes = user_orders.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
        
        # User-level basket features
        basket_features = basket_sizes.groupby('user_id')['basket_size'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        
        basket_features.columns = [
            'avg_basket_size', 'std_basket_size', 'min_basket_size',
            'max_basket_size', 'median_basket_size'
        ]
        
        basket_features = basket_features.fillna(0)
        
        return basket_features.reset_index()
    
    def _build_product_pattern_features(self, user_orders: pd.DataFrame) -> pd.DataFrame:
        """Build product-level pattern features"""
        
        # Product diversity
        product_diversity = user_orders.groupby('user_id')['product_id'].agg([
            'nunique'
        ]).rename(columns={'nunique': 'unique_products'})
        
        # Reorder patterns
        reorder_patterns = user_orders.groupby('user_id').agg({
            'reordered': ['mean', 'sum'],
            'add_to_cart_order': ['mean', 'std']
        }).round(4)
        
        reorder_patterns.columns = [
            'reorder_rate', 'total_reorders', 'avg_cart_position', 'std_cart_position'
        ]
        
        # Combine features
        product_features = product_diversity.merge(reorder_patterns, on='user_id')
        product_features = product_features.fillna(0)
        
        return product_features.reset_index()
    
    def _build_department_aisle_features(self, user_orders: pd.DataFrame) -> pd.DataFrame:
        """Build department and aisle preference features"""
        
        # Department preferences
        dept_prefs = user_orders.groupby(['user_id', 'department']).size().reset_index(name='dept_count')
        dept_diversity = dept_prefs.groupby('user_id')['department'].nunique().reset_index(name='unique_departments')
        
        # Top department for each user
        top_dept = dept_prefs.loc[dept_prefs.groupby('user_id')['dept_count'].idxmax()][['user_id', 'department']]
        top_dept = top_dept.rename(columns={'department': 'top_department'})
        
        # Aisle preferences
        aisle_prefs = user_orders.groupby(['user_id', 'aisle']).size().reset_index(name='aisle_count')  
        aisle_diversity = aisle_prefs.groupby('user_id')['aisle'].nunique().reset_index(name='unique_aisles')
        
        # Top aisle for each user
        top_aisle = aisle_prefs.loc[aisle_prefs.groupby('user_id')['aisle_count'].idxmax()][['user_id', 'aisle']]
        top_aisle = top_aisle.rename(columns={'aisle': 'top_aisle'})
        
        # Merge department/aisle features
        dept_aisle_features = dept_diversity.merge(top_dept, on='user_id')
        dept_aisle_features = dept_aisle_features.merge(aisle_diversity, on='user_id') 
        dept_aisle_features = dept_aisle_features.merge(top_aisle, on='user_id')
        
        return dept_aisle_features