# Instacart EDA & RFM Analysis Summary

## Overview
This document summarizes the Exploratory Data Analysis and RFM customer segmentation performed on the Instacart dataset (~3M orders from 206K users).

## Dataset Statistics
- **Users**: 206,209 unique customers
- **Orders**: 3,421,083 total orders
- **Order Items**: 33,819,106 individual product orders
- **Products**: 49,688 unique products
- **Departments**: 21 product categories
- **Aisles**: 134 product subcategories

## Key Behavioral Insights

### Temporal Patterns
- **Peak Hour**: 10:00 AM (288,418 orders)
- **Popular Day**: Sunday (600,905 orders)
- **Pattern**: Morning shopping (9-15h) preferred, weekend peaks

### Order Characteristics
- **Average Order Size**: 10.11 items
- **Median Order Size**: 8 items
- **Reorder Rate**: 59% (high customer loyalty)
- **Max Orders per User**: 100 orders

### Product Preferences
1. **Fresh Fruits**: 3,792,661 orders
2. **Fresh Vegetables**: 3,568,630 orders  
3. **Packaged Vegetables/Fruits**: 1,843,806 orders
4. **Yogurt**: 1,507,583 orders
5. **Packaged Cheese**: 1,021,462 orders

**Top Department**: Produce (9.9M orders, 29.3% of all orders)

## RFM Customer Segmentation

### Segment Distribution
| Segment | Count | Percentage | Description |
|---------|-------|------------|-------------|
| Champions | 47,277 | 22.9% | Best customers - recent, frequent, high value |
| Lost | 41,281 | 20.0% | Previously active, now inactive |
| About to Sleep | 32,178 | 15.6% | Declining engagement, at risk |
| Loyal Customers | 21,351 | 10.4% | Consistent, loyal base |
| Potential Loyalists | 18,168 | 8.8% | Recent customers with potential |
| Promising | 15,105 | 7.3% | New customers showing promise |
| Others | 30,849 | 15.0% | Various smaller segments |

## Business Recommendations

### High Priority Actions
1. **Champion Retention**: VIP programs for 47K top customers
2. **Win-back Campaigns**: Target 41K lost customers  
3. **Churn Prevention**: Proactive engagement for 32K "about to sleep"

### Operational Optimizations
- **Inventory**: Focus on fresh produce for weekend/morning demand
- **Marketing**: Time campaigns for Sunday/morning shopping windows
- **Product Strategy**: Leverage 59% reorder rate for subscriptions

## Output Files
- `data/processed/users_rfm.parquet` - RFM analysis results
- `data/processed/products_enriched.parquet` - Product hierarchy data

## Next Steps
1. **Feature Engineering** (Notebook 02) - User/item/interaction features
2. **Baseline Models** (Notebook 03) - XGBoost recommendation baseline
3. **Deep Learning** (Notebook 04) - LSTM/TCN sequence models

---
