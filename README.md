## Movie Recommendation and Rating Prediction


This project uses the MovieLens `ml-latest-small` dataset to develop, compare, and **fuse** various algorithms for movie recommendation and rating prediction. We evaluate seven base methods — Singular Value Decomposition (SVD), content-based filtering, item-to-item K-Nearest Neighbors (KNN), user-based filtering, neural collaborative filtering (NeuMF, He et al. 2017), and Personalized PageRank on the bipartite user-movie graph — and combine them through a **gradient-boosted regression tree (LightGBM) stacking layer** adapted from Volokhin et al. (NAACL 2021). Performance metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) are used alongside ranking metrics (HR@10, NDCG@10, Precision/Recall/F1@10).


### Problem Statement

Data Preprocessing:

1. Create training, validation, and test datasets using **leave-one-out by timestamp** (He et al., 2017): for each user, the most recent rating becomes the test point, the second-most-recent becomes the validation point, and the rest becomes training. This replaces a naive random 80/20 split, which would leak future information into the training set.

Rating Prediction:

2. Develop algorithms to predict the ratings in the testing set based on the information in the training dataset.

3. Evaluate the predictions based on Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Item Recommendation:

1. Construct a recommendation list for each user using a sampled-negatives protocol — for every test interaction, mix the true held-out item with 99 randomly sampled unseen items and rank the resulting 100 candidates.

2. Evaluate the recommendation quality based on Hit Ratio (HR@10), Normalized Discounted Cumulative Gain (NDCG@10), Precision@10, Recall@10, and F1@10, with the Krichene & Rendle (2020) caveat on sampled-metric limitations noted in the report.

### Data and Exploratory Data Analysis

MovieLens `ml-latest-small`: **100,836 ratings**, **9,742 movies**, **610 users**, rating scale 0.5–5.0 in half-star increments, ~98% sparse. Released 09/2018 on https://grouplens.org/datasets/movielens/. We chose this dataset for tractability on free-tier Google Colab; the methodology generalizes to larger MovieLens variants (10M, 25M) without architectural change. Below figures show the distribution of ratings per movie, per user, and per genre.

![image](https://github.com/user-attachments/assets/cdd64d2f-397c-41ec-8d60-11d4f2703d09)

![image](https://github.com/user-attachments/assets/0e6970b6-0f8b-4343-ba06-7287cfe28ee9)

![image](https://github.com/user-attachments/assets/012a30ab-8bbf-4fcb-b8c6-75e3bffb12f4)

### Hybrid Fusion Model

The project's headline contribution is a LightGBM meta-learner that takes each baseline's predicted rating plus seven side features (user mean rating, user rating count, item mean rating, item rating count, item popularity bin, item age, content similarity) as input — 15 features in total — and produces a single fused rating. To avoid leakage when training the meta-learner, we use **3-fold out-of-fold cross-prediction**: every training row's base predictions come from models that did not see that row during training, ensuring the meta-learner sees realistic predictions rather than memorized ones. The architecture is adapted from Volokhin et al. (NAACL 2021).

### Results

The hybrid GBRT achieves the lowest RMSE (0.9448) and MAE (0.7277), outperforming all eight individual baselines on rating prediction; paired Wilcoxon signed-rank tests on per-user NDCG@10 confirm the gains are statistically significant (p < 0.01) against most baselines. The content-based approach showed higher RMSE compared to collaborative filtering methods, item-item collaborative filtering outperformed user-user filtering due to the dataset's larger number of movies relative to users, and SVD and KNN-Baseline fell in between. On **ranking metrics**, however, Personalized PageRank dominates (HR@10 = 0.6623, NDCG@10 = 0.4135) — substantially exceeding the hybrid's HR@10 = 0.1148. This reflects the well-documented rating-vs-ranking mismatch (Cremonesi et al., 2010): LightGBM trained with squared-error loss collapses toward predicting item means, which is calibrated for RMSE but does not differentiate per-user enough for ranking. Stratified cold-start analysis (binning by user activity × item popularity) confirms the hybrid's gains concentrate on heavy-user / popular-item regions; in the cold-cold corner, single-method baselines outperform it. These findings highlight the importance of aligning the meta-learner's loss function with the target evaluation metric.

![image](https://github.com/user-attachments/assets/a5e6c4b9-3897-4a9d-8b65-4611d5843029)

![image](https://github.com/user-attachments/assets/a34ff0cd-2080-4219-b5a0-760148f13c8b)

**Attribution.** GBRT-fusion architecture adapted from Volokhin, Pera, Tintarev (NAACL 2021). Implementation pattern referenced from Mehta, Parikh, and Gaglani's `movie-rec-conversational-data` repository (included in this project's subfolder). NeuMF follows He et al. (WWW 2017).


