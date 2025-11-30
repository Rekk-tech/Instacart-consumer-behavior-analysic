# Feature Engineering Documentation

## Overview
This document describes all engineered features for the Instacart Next Purchase Prediction project.

## Feature Categories

### 1. User Features (37 features, 206K users)
Location: `data/features/user_features_with_rfm.parquet`

#### Temporal Behavior (7 features)
- `user_total_orders`: Total number of orders placed
- `user_avg_dow`: Average day of week (0=Sunday, 6=Saturday)  
- `user_std_dow`: Standard deviation of ordering days
- `user_unique_dow`: Number of unique days user has ordered
- `user_avg_hour`: Average hour of day for orders
- `user_std_hour`: Standard deviation of ordering hours
- `user_unique_hours`: Number of unique hours user has ordered

#### Ordering Frequency (4 features)
- `user_avg_days_between_orders`: Average days between consecutive orders
- `user_std_days_between_orders`: Standard deviation of inter-order days
- `user_min_days_between_orders`: Minimum days between orders
- `user_max_days_between_orders`: Maximum days between orders

#### Basket Behavior (5 features)
- `user_avg_basket_size`: Average number of items per order
- `user_std_basket_size`: Standard deviation of basket sizes
- `user_min_basket_size`: Minimum basket size
- `user_max_basket_size`: Maximum basket size  
- `user_median_basket_size`: Median basket size

#### Product Diversity (6 features)
- `user_total_products`: Total products purchased across all orders
- `user_unique_products`: Number of unique products purchased
- `user_avg_products_per_order`: Average products per order
- `user_product_discovery_rate`: Ratio of unique to total products
- `user_unique_departments`: Number of departments shopped
- `user_unique_aisles`: Number of aisles shopped

#### Reorder Behavior (6 features)  
- `user_reorder_rate`: Percentage of items that are reorders
- `user_total_reorders`: Total number of reordered items
- `user_total_order_items`: Total items across all orders
- `user_reorder_ratio`: Reorders / total items ratio
- `user_dept_exploration`: Departments explored / 21 total
- `user_aisle_exploration`: Aisles explored / 134 total

#### RFM Segmentation (9 features)
- `recency`: Average days between orders (lower = better)
- `frequency`: Total number of orders
- `monetary`: Total products purchased (proxy for spend)
- `R_score`: Recency score (1-5, higher = better)
- `F_score`: Frequency score (1-5, higher = better)
- `M_score`: Monetary score (1-5, higher = better)
- `RFM_score`: Combined RFM score string (e.g., "543")
- `Segment`: Customer segment (Champions, Loyal, At Risk, etc.)

### 2. Item Features (25 features, 50K products)
Location: `data/features/item_features.parquet`

#### Basic Product Info (5 features)
- `product_name`: Product name
- `aisle_id`: Aisle identifier  
- `department_id`: Department identifier
- `aisle`: Aisle name
- `department`: Department name

#### Popularity Metrics (6 features)
- `item_total_orders`: Total times this product was ordered
- `item_unique_orders`: Number of unique orders containing this product
- `item_unique_users`: Number of unique users who bought this product
- `item_user_penetration`: Percentage of total users who bought this
- `item_dept_popularity_rank`: Rank within department by orders
- `item_aisle_popularity_rank`: Rank within aisle by orders

#### Cart Position Behavior (4 features)
- `item_avg_cart_position`: Average position in cart when added
- `item_std_cart_position`: Standard deviation of cart positions
- `item_min_cart_position`: Earliest cart position
- `item_max_cart_position`: Latest cart position

#### Reorder Analysis (5 features)
- `item_reorder_rate`: Percentage of orders where this was reordered
- `item_total_reorders`: Total number of reorders
- `item_reorder_probability`: Reorders / total orders ratio
- `item_vs_dept_reorder_rate`: Item reorder rate - department average
- `item_vs_aisle_reorder_rate`: Item reorder rate - aisle average

#### Category Statistics (5 features)
- `dept_total_orders`: Total orders in this product's department
- `dept_avg_reorder_rate`: Average reorder rate for department
- `aisle_total_orders`: Total orders in this product's aisle
- `aisle_avg_reorder_rate`: Average reorder rate for aisle

### 3. User-Item Interaction Features (14 features, 13.9M interactions)
Location: `data/features/user_item_features.parquet`

#### Purchase History (3 features)
- `ui_times_bought`: Number of times user bought this product
- `ui_times_reordered`: Number of times user reordered this product
- `ui_reorder_rate`: User's reorder rate for this specific product

#### Cart Behavior (3 features)
- `ui_avg_cart_position`: User's average cart position for this product
- `ui_std_cart_position`: Standard deviation of cart positions
- `ui_min_cart_position`: Earliest cart position for this product

#### Temporal Patterns (6 features)
- `ui_first_order_number`: User's order number when first bought
- `ui_last_order_number`: User's order number when last bought  
- `ui_avg_order_number`: Average order number when buying this product
- `ui_order_span`: Orders between first and last purchase
- `ui_purchase_frequency`: Purchase frequency within span
- `ui_orders_since_last_purchase`: Orders since last buying this product

## Data Quality Notes

### Missing Values
- All features have been cleaned and filled with appropriate defaults (0 for most metrics)
- Single-order users have NaN for inter-order statistics (filled with 0 or defaults)
- Products never purchased by a user won't appear in user-item features

### Feature Scaling
- Features are raw values - scaling may be needed for some models
- Rates and ratios are bounded [0,1] or similar ranges
- Counts can have high variance (e.g., total_orders: 1-100)

### Data Types
- Categorical: `Segment`, `department`, `aisle`, `product_name`
- Continuous: Most other features (floats)
- Discrete counts: `*_total_*`, `*_unique_*`, `*_times_*`

## Usage Notes

### For Model Training
1. Join features as needed for your candidate generation strategy
2. Consider feature selection based on correlation analysis
3. Apply appropriate scaling for algorithms that need it
4. Handle categorical encoding (e.g., one-hot, label encoding)

### Feature Importance Expected
- High importance: `ui_times_bought`, `ui_reorder_rate`, `item_reorder_rate`
- Medium importance: User RFM scores, item popularity metrics
- Lower importance: Variance/std features, exploration ratios

### Memory Considerations
- User-item features are largest (13.9M rows) - consider sampling for prototyping
- Use parquet format for efficient loading
- Consider chunked processing for large model training

---
*Generated from 02_feature_engineering.ipynb*