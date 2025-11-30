"""
Metrics calculation utilities for Instacart recommendation system.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)
import logging

logger = logging.getLogger(__name__)

def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
    }
    
    return metrics

def calculate_ranking_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Calculate ranking metrics (Precision@K, Recall@K)."""
    
    metrics = {}
    
    for k in k_values:
        # Precision@K
        top_k_idx = np.argsort(y_pred_proba)[-k:]
        precision_at_k = y_true[top_k_idx].mean()
        metrics[f'precision@{k}'] = precision_at_k
        
        # Recall@K  
        recall_at_k = y_true[top_k_idx].sum() / y_true.sum() if y_true.sum() > 0 else 0
        metrics[f'recall@{k}'] = recall_at_k
        
        # F1@K
        if precision_at_k + recall_at_k > 0:
            f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        else:
            f1_at_k = 0
        metrics[f'f1@{k}'] = f1_at_k
    
    return metrics

def calculate_recommendation_metrics(
    recommendations: Dict[int, List[int]], 
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Calculate recommendation-specific metrics."""
    
    metrics = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        
        for user_id in recommendations.keys():
            if user_id not in ground_truth:
                continue
                
            rec_k = recommendations[user_id][:k]
            true_items = set(ground_truth[user_id])
            rec_items = set(rec_k)
            
            # Precision@K for this user
            if len(rec_items) > 0:
                precision = len(rec_items.intersection(true_items)) / len(rec_items)
            else:
                precision = 0
            precisions.append(precision)
            
            # Recall@K for this user  
            if len(true_items) > 0:
                recall = len(rec_items.intersection(true_items)) / len(true_items)
            else:
                recall = 0
            recalls.append(recall)
        
        metrics[f'avg_precision@{k}'] = np.mean(precisions)
        metrics[f'avg_recall@{k}'] = np.mean(recalls)
        
        # Average F1@K
        avg_prec = metrics[f'avg_precision@{k}']
        avg_rec = metrics[f'avg_recall@{k}']
        if avg_prec + avg_rec > 0:
            metrics[f'avg_f1@{k}'] = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
        else:
            metrics[f'avg_f1@{k}'] = 0
    
    return metrics

def calculate_coverage_metrics(
    recommendations: Dict[int, List[int]],
    all_items: List[int]
) -> Dict[str, float]:
    """Calculate catalog coverage and diversity metrics."""
    
    # Get all recommended items
    all_recommended = set()
    for user_recs in recommendations.values():
        all_recommended.update(user_recs)
    
    # Catalog coverage
    catalog_coverage = len(all_recommended) / len(all_items)
    
    # Average list diversity (intra-list diversity)
    diversities = []
    for user_recs in recommendations.values():
        if len(user_recs) > 1:
            # Simple diversity: number of unique items / total items
            diversity = len(set(user_recs)) / len(user_recs)
            diversities.append(diversity)
    
    avg_diversity = np.mean(diversities) if diversities else 0
    
    return {
        'catalog_coverage': catalog_coverage,
        'avg_list_diversity': avg_diversity,
        'total_unique_recommendations': len(all_recommended)
    }

def calculate_business_metrics(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, float],
    item_revenue: Optional[Dict[int, float]] = None
) -> Dict[str, float]:
    """Calculate business-relevant metrics."""
    
    metrics = {}
    
    # Average popularity of recommendations
    popularities = []
    for user_recs in recommendations.values():
        user_popularities = [item_popularity.get(item, 0) for item in user_recs]
        popularities.extend(user_popularities)
    
    metrics['avg_item_popularity'] = np.mean(popularities) if popularities else 0
    
    # Revenue metrics (if available)
    if item_revenue:
        revenues = []
        for user_recs in recommendations.values():
            user_revenues = [item_revenue.get(item, 0) for item in user_recs]
            revenues.extend(user_revenues)
        
        metrics['avg_item_revenue'] = np.mean(revenues) if revenues else 0
        metrics['total_potential_revenue'] = sum(revenues)
    
    return metrics

def create_metrics_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray, 
    model_name: str,
    additional_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Create comprehensive metrics report."""
    
    report = {
        'model_name': model_name,
        'data_stats': {
            'total_samples': len(y_true),
            'positive_samples': int(y_true.sum()),
            'positive_rate': float(y_true.mean()),
            'negative_samples': int(len(y_true) - y_true.sum())
        }
    }
    
    # Classification metrics
    report['classification'] = calculate_classification_metrics(y_true, y_pred_proba)
    
    # Ranking metrics
    report['ranking'] = calculate_ranking_metrics(y_true, y_pred_proba)
    
    # Additional metrics
    if additional_metrics:
        report['additional'] = additional_metrics
    
    # Improvement over random
    random_auc = 0.5
    auc_improvement = (report['classification']['auc'] - random_auc) / random_auc * 100
    report['improvement_over_random'] = f"{auc_improvement:.1f}%"
    
    return report

def print_metrics_summary(metrics_report: Dict[str, Any]) -> None:
    """Print formatted metrics summary."""
    
    print(f"\n{'='*60}")
    print(f"📊 METRICS REPORT: {metrics_report['model_name']}")  
    print(f"{'='*60}")
    
    # Data stats
    stats = metrics_report['data_stats']
    print(f"\n📈 Data Statistics:")
    print(f"    Total samples: {stats['total_samples']:,}")
    print(f"    Positive samples: {stats['positive_samples']:,} ({stats['positive_rate']:.2%})")
    print(f"    Negative samples: {stats['negative_samples']:,}")
    
    # Classification metrics
    cls_metrics = metrics_report['classification']
    print(f"\n🎯 Classification Metrics:")
    print(f"    AUC: {cls_metrics['auc']:.4f}")
    print(f"    Accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"    Precision: {cls_metrics['precision']:.4f}")
    print(f"    Recall: {cls_metrics['recall']:.4f}")
    print(f"    F1-Score: {cls_metrics['f1']:.4f}")
    
    # Ranking metrics
    ranking_metrics = metrics_report['ranking']
    print(f"\n🏆 Ranking Metrics:")
    for metric, value in ranking_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Improvement
    print(f"\n📈 Performance:")
    print(f"    Improvement over random: {metrics_report['improvement_over_random']}")
    
    print(f"\n{'='*60}")

def save_metrics_report(
    metrics_report: Dict[str, Any], 
    output_path: str
) -> None:
    """Save metrics report to JSON file."""
    
    import json
    from pathlib import Path
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(v) for v in d]
        else:
            return convert_numpy(d)
    
    clean_report = clean_dict(metrics_report)
    
    with open(output_path, 'w') as f:
        json.dump(clean_report, f, indent=2)
    
    logger.info(f"Metrics report saved to {output_path}")

class RecommendationMetrics:
    """Recommendation evaluation metrics"""
    
    @staticmethod
    def precision_at_k(predictions: np.ndarray, k: int) -> float:
        """Calculate Precision@K"""
        if len(predictions) == 0:
            return 0.0
        return np.mean(predictions[:k]) if k <= len(predictions) else np.mean(predictions)
    
    @staticmethod  
    def recall_at_k(predictions: np.ndarray, total_relevant: np.ndarray, k: int) -> float:
        """Calculate Recall@K"""
        if len(total_relevant) == 0:
            return 0.0
        relevant_at_k = np.sum(predictions[:k]) if k <= len(predictions) else np.sum(predictions)
        return relevant_at_k / np.sum(total_relevant) if np.sum(total_relevant) > 0 else 0.0

__all__ = [
    'calculate_classification_metrics',
    'calculate_ranking_metrics', 
    'calculate_recommendation_metrics',
    'calculate_coverage_metrics',
    'calculate_business_metrics',
    'RecommendationMetrics'
]