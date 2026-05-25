## Movie Recommendation and Rating Prediction


This project uses the MovieLens dataset to develop and compare various algorithms for movie recommendation and rating prediction. We evaluate the performance of different approaches including Singular Value Decomposition (SVD), content-based filtering, item-to-item K-Nearest Neighbors (KNN), user-based filtering, hybrid methods, and neural filtering. Performance metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) are used to assess the effectiveness of each algorithm. Our comprehensive analysis provides insights into the strengths and weaknesses of each method, helping to identify the best strategies for accurate movie recommendations.


### Problem Statement

Data Preprocessing:

1. Create a training dataset and a testing dataset for the experiment.
For each user, randomly select 80% of their ratings as the training ratings, and use the remaining 20% as testing ratings.
The training ratings from all users consist of the final training dataset, and the testing ratings from all users consist of the final testing dataset.
Rating Prediction:

2. Develop algorithms to predict the ratings in the testing set based on the information in the training dataset.

3. Evaluate the predictions based on Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Item Recommendation:

1. Construct a recommendation list for each user.

2. Evaluate the recommendation quality based on precision, recall, F-measure, and Normalized Discounted Cumulative Gain (NDCG).

### Data and Exploratory Data Analysis

25 million ratings applied to 62,000 movies by 162,000 users. Released 12/2019 on https://grouplens.org/datasets/movielens/. Below figure shows the distribution of ratings per movie. Of the 2,83,228 users using Movies, around 69,000 people rated more then 100 movies. Most of the movies in the data received less than 10000 ratings.

![image](https://github.com/user-attachments/assets/cdd64d2f-397c-41ec-8d60-11d4f2703d09)

![image](https://github.com/user-attachments/assets/0e6970b6-0f8b-4343-ba06-7287cfe28ee9)

![image](https://github.com/user-attachments/assets/012a30ab-8bbf-4fcb-b8c6-75e3bffb12f4)

### Results

The study's results show significant differences in performance among various recommendation algorithms evaluated on the MovieLens dataset. Specifically, the content-based approach showed higher RMSE and MAE values compared to collaborative filtering methods, with item-item collaborative filtering outperforming user-user filtering due to the dataset's larger number of movies relative to users. KNN-based filtering fell in between these two approaches. The most successful models were SVD-based and neural collaborative filtering, which achieved the lowest error rates and higher precision and recall metrics. This success highlights the advantage of collaborative filtering. However, on the other hand, the content-based method's reliance only on movie genres limited its effectiveness. The superior performance of SVD and neural models is explained by the fact that these models capture latent factors in user-movie interactions, with the neural model excelling in capturing non-linear relationships. These findings highlight the importance of algorithm choice and dataset characteristics in optimizing recommendation system performance.

![image](https://github.com/user-attachments/assets/a5e6c4b9-3897-4a9d-8b65-4611d5843029)

![image](https://github.com/user-attachments/assets/a34ff0cd-2080-4219-b5a0-760148f13c8b)

