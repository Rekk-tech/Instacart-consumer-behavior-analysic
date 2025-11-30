"""
ETL module for RFM analysis.
Copy logic from notebook 01 (EDA) for RFM segmentation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

from src.utils import get_logger, load_parquet, save_parquet, ensure_directories

logger = get_logger(__name__)

class RFMProcessor:
    """RFM (Recency, Frequency, Monetary) analysis processor."""
    
    def __init__(self, processed_data_path: str, features_data_path: str, quantiles: int = 5):
        self.processed_data_path = processed_data_path
        self.features_data_path = features_data_path
        self.quantiles = quantiles
        
        # Ensure output directory exists
        ensure_directories(features_data_path)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load processed data for RFM analysis."""
        
        logger.info("Loading data for RFM analysis...")
        
        data = {}
        required_tables = ['orders', 'order_products_prior', 'order_products_train', 'products']
        
        for table in required_tables:
            file_path = f"{self.processed_data_path}/{table}.parquet"
            data[table] = load_parquet(file_path)
            logger.info(f"Loaded {table}: {data[table].shape}")
        
        return data
    
    def prepare_order_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare consolidated order data."""
        
        logger.info("Preparing consolidated order data...")
        
        # Combine prior and train order products
        order_products = pd.concat([
            data['order_products_prior'],
            data['order_products_train'] 
        ], ignore_index=True)
        
        logger.info(f"Combined order products: {order_products.shape}")
        
        # Merge with orders to get user and order info
        orders_full = data['orders'].merge(
            order_products, 
            on='order_id', 
            how='inner'
        )
        
        logger.info(f"Orders with products: {orders_full.shape}")
        
        # Add product information
        orders_with_products = orders_full.merge(
            data['products'][['product_id', 'department_id', 'aisle_id']], 
            on='product_id',
            how='left'
        )
        
        logger.info(f"Orders with product info: {orders_with_products.shape}")
        
        return orders_with_products
    
    def calculate_rfm_metrics(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM metrics for each user."""
        
        logger.info("Calculating RFM metrics...")
        
        # Get the maximum order number as reference point for recency
        max_order_number = orders_df['order_number'].max()
        logger.info(f"Max order number: {max_order_number}")
        
        # Calculate user-level aggregations
        user_metrics = orders_df.groupby('user_id').agg({
            'order_number': ['max', 'nunique', 'min'],  # For recency and frequency
            'product_id': 'count',  # Total items purchased (monetary proxy)
            'order_id': 'nunique',  # Number of unique orders
            'reordered': 'mean',  # Reorder rate
            'days_since_prior_order': 'mean'  # Average days between orders
        }).reset_index()
        
        # Flatten column names
        user_metrics.columns = [
            'user_id', 'max_order_number', 'unique_order_numbers', 'min_order_number',
            'total_products', 'unique_orders', 'reorder_rate', 'avg_days_between_orders'
        ]
        
        # Calculate RFM components
        # Recency: How recently did the customer make a purchase? (lower is better)
        # Use inverse of max_order_number so higher values = more recent
        user_metrics['recency'] = max_order_number - user_metrics['max_order_number'] + 1
        
        # Frequency: How often do they purchase? (higher is better)
        user_metrics['frequency'] = user_metrics['unique_orders']
        
        # Monetary: How much do they spend? (use total products as proxy)
        user_metrics['monetary'] = user_metrics['total_products']
        
        logger.info(f"Calculated RFM metrics for {len(user_metrics)} users")
        
        return user_metrics
    
    def create_rfm_scores(self, user_metrics: pd.DataFrame) -> pd.DataFrame:
        """Create RFM scores using quantiles."""
        
        logger.info(f"Creating RFM scores with {self.quantiles} quantiles...")
        
        rfm_data = user_metrics.copy()
        
        # Calculate quantile-based scores (1 = worst, quantiles = best)
        # For Recency: lower values are better (more recent)
        rfm_data['R_score'] = pd.qcut(
            rfm_data['recency'], 
            q=self.quantiles, 
            labels=range(self.quantiles, 0, -1),  # Reverse for recency
            duplicates='drop'
        ).astype(int)
        
        # For Frequency: higher values are better
        rfm_data['F_score'] = pd.qcut(
            rfm_data['frequency'],
            q=self.quantiles,
            labels=range(1, self.quantiles + 1),
            duplicates='drop'
        ).astype(int)
        
        # For Monetary: higher values are better  
        rfm_data['M_score'] = pd.qcut(
            rfm_data['monetary'],
            q=self.quantiles, 
            labels=range(1, self.quantiles + 1),
            duplicates='drop'
        ).astype(int)
        
        # Combined RFM score
        rfm_data['RFM_score'] = (
            rfm_data['R_score'].astype(str) + 
            rfm_data['F_score'].astype(str) + 
            rfm_data['M_score'].astype(str)
        )
        
        logger.info("RFM scores calculated successfully")
        
        return rfm_data
    
    def create_rfm_segments(self, rfm_data: pd.DataFrame) -> pd.DataFrame:
        """Create customer segments based on RFM scores."""
        
        logger.info("Creating RFM customer segments...")
        
        # Define segmentation logic
        def assign_segment(row):
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            
            # Champions: High RFM
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            # Loyal Customers: High RF, medium M
            elif r >= 4 and f >= 4 and m >= 2:
                return 'Loyal_Customers'
            # Potential Loyalists: High R, medium F
            elif r >= 4 and f >= 2 and m >= 2:
                return 'Potential_Loyalists'
            # New Customers: High R, low F
            elif r >= 4 and f <= 2:
                return 'New_Customers'
            # Promising: Medium R, low F, high M  
            elif r >= 2 and f <= 2 and m >= 4:
                return 'Promising'
            # Need Attention: Medium RF, high M
            elif r >= 2 and f >= 2 and m >= 4:
                return 'Need_Attention'
            # About to Sleep: Low R, medium F
            elif r <= 2 and f >= 2 and m >= 2:
                return 'About_to_Sleep'
            # At Risk: Low R, high F, high M
            elif r <= 2 and f >= 4 and m >= 4:
                return 'At_Risk'
            # Cannot Lose Them: Low R, high F, very high M
            elif r <= 2 and f >= 4 and m >= 4:
                return 'Cannot_Lose_Them'
            # Hibernating: Low RFM
            else:
                return 'Hibernating'
        
        rfm_segments = rfm_data.copy()
        rfm_segments['Segment'] = rfm_segments.apply(assign_segment, axis=1)
        
        # Log segment distribution
        segment_counts = rfm_segments['Segment'].value_counts()
        logger.info("RFM Segment distribution:")
        for segment, count in segment_counts.items():
            logger.info(f"  {segment}: {count:,} ({count/len(rfm_segments)*100:.1f}%)")
        
        return rfm_segments
    
    def save_rfm_analysis(self, rfm_segments: pd.DataFrame) -> None:
        """Save RFM analysis results."""
        
        logger.info("Saving RFM analysis results...")
        
        # Save full RFM data
        output_path = f"{self.features_data_path}/rfm_analysis.parquet"
        save_parquet(rfm_segments, output_path, index=False)
        
        # Save segment mapping for easy lookup
        segment_mapping = rfm_segments[['user_id', 'Segment', 'RFM_score', 'R_score', 'F_score', 'M_score']]
        segment_path = f"{self.features_data_path}/user_segments.parquet"
        save_parquet(segment_mapping, segment_path, index=False)
        
        logger.info("RFM analysis saved successfully")
    
    def process(self) -> pd.DataFrame:
        """Main RFM processing pipeline."""
        
        logger.info("Starting RFM analysis pipeline...")
        
        # Load data
        data = self.load_data()
        
        # Prepare order data
        orders_df = self.prepare_order_data(data)
        
        # Calculate RFM metrics
        user_metrics = self.calculate_rfm_metrics(orders_df)
        
        # Create RFM scores
        rfm_data = self.create_rfm_scores(user_metrics)
        
        # Create segments
        rfm_segments = self.create_rfm_segments(rfm_data)
        
        # Save results
        self.save_rfm_analysis(rfm_segments)
        
        logger.info("✅ RFM analysis completed successfully")
        return rfm_segments

def process_rfm_analysis(
    processed_data_path: str,
    features_data_path: str, 
    quantiles: int = 5
) -> pd.DataFrame:
    """Main function to run RFM analysis."""
    
    processor = RFMProcessor(processed_data_path, features_data_path, quantiles)
    return processor.process()

if __name__ == "__main__":
    # Example usage
    from src.utils import setup_logging
    
    setup_logging()
    
    # Run RFM analysis
    rfm_results = process_rfm_analysis(
        processed_data_path="data/processed",
        features_data_path="data/features"
    )
    
    print(f"RFM analysis completed for {len(rfm_results)} users")
    print("\nSample results:")
    print(rfm_results.head())