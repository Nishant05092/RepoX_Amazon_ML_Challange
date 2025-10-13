AMAZON ML CHALLENGE 2025

Team Name: RepoX
Team Members: Nishant Sharma (Leader), Anurag Jaiswal, Parth Tiwari, Sumit Pathak
Submission Date: 13-10-2025

Executive Summary

Our approach integrates multimodal learning by combining textual and visual information to predict product prices.
CLIP ViT-B/32 generates embeddings for both product descriptions and images, which are fused using a gated fusion mechanism.
The fused embeddings feed into an XGBoost regressor, enabling robust price prediction across diverse product categories.

Problem Analysis

Task: Multimodal regression to predict product prices using text and image data.

Insights from EDA:

Price distribution is highly skewed with outliers → log transformation applied.

Text features (brand, quantity, specifications) strongly correlate with price.

Image features provide additional visual cues; resizing and normalization are essential.

Approach: Hybrid multimodal learning using CLIP embeddings + gated fusion.

Core Innovation:

Uses a shared embedding space for text and images.

Gated fusion dynamically weights modalities based on informativeness.

Nonlinear regression with XGBoost handles skewed distribution and outliers better than linear or other tree-based models.

Architecture Overview

Text and images → CLIP ViT-B/32 embeddings (512-D) → gated fusion → optional PCA → XGBoost Regressor predicts log-transformed prices.

Modular, end-to-end pipeline with normalization and missing-image handling.

Model Components
Text Processing

Clean text, remove artifacts, HTML entities, prefixes, normalize whitespace.

Fill missing text with empty strings.

Text Encoder: CLIP ViT-B/32 (512-D), L2-normalized, fused with image embeddings via gated fusion.

Image Processing

Load images safely, convert to RGB, resize to 224×224, normalize pixels.

Replace missing/corrupt images with black placeholders.

Image Encoder: CLIP ViT-B/32 (512-D), L2-normalized, fused with text embeddings via gated fusion.

Model Performance Metrics

SMAPE: 0.523

MAE: 0.592566

RMSE: 0.755101

R²: 0.354190

Conclusion

We developed a robust multimodal regression pipeline combining text and image features via CLIP embeddings and gated fusion.
XGBoost regressor predicts log-transformed prices effectively, handling skewness and outliers. 
The modular pipeline ensures scalability, normalization, and missing data handling, achieving strong generalization across product categories.
