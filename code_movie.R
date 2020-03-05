# First load packages
library(ggplot2)
library(tidyverse)
library(kableExtra)
library(recosystem)
library(data.table)

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
# set.seed(1, sample.kind="Rounding")
set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


edx %>% class()
dim(edx)
head(edx)

edx %>% 
    summarise(users = n_distinct(userId),
              movies = n_distinct(movieId))
edx %>%
    select(genres) %>% 
    separate_rows(genres, sep = "\\|") %>% 
    pull(genres) %>% unique()

# check any missing data
any(is.na(edx))

edx_movies <- 
    edx %>% 
    group_by(movieId) %>% 
    summarise(count = n()) %>% 
    arrange(-count)
edx_movies %>% summary()

# What are the most rated movies?
edx %>% 
    group_by(title) %>% 
    summarise(count = n()) %>% 
    arrange(-count) %>% 
    top_n(10) %>% 
    ggplot(aes(x = reorder(title, -count), y = count)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0, hjust = 1)) +
    xlab("Top 10 most rated movies")

# What are the best rated movies?
edx %>% 
    group_by(title) %>% 
    filter(n() >= 1000) %>%  
    summarise(avg = mean(rating), median = median(rating), n = n()) %>% 
    arrange(-avg) %>% 
    top_n(10) %>% 
    ggplot(aes(reorder(title, -avg), avg)) +
    geom_bar(stat = "identity") +
    xlab("") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0, hjust = 1, size = 15)) +
    ylab("Average rating") + ggtitle("Top 10 best rated movies which at least received 1000 ratings")

# What are the most rated genres?
edx %>% 
    separate_rows(genres, sep = "\\|") %>% 
    group_by(genres) %>% 
    summarise(count = n()) %>% 
    arrange(-count)

edx %>% 
    separate_rows(genres, sep = "\\|") %>% 
    group_by(genres) %>% 
    summarise(count = n()) %>% 
    ggplot(aes(reorder(genres, -count), count)) +
    geom_bar(stat = "identity") +
    xlab("") +
    theme(axis.text.x = element_text(size = 15, angle = 90, hjust = 1, vjust = 0)) +
    ggtitle("Number of rated genres")

# What are the best rated genres?
edx %>% 
    separate_rows(genres, sep = "\\|") %>% 
    group_by(genres) %>% 
    summarise(avg = mean(rating), median = median(rating), count = n()) %>% 
    ggplot(aes(reorder(genres, -avg), avg)) +
    geom_bar(stat = "identity") +
    xlab("") +
    theme(axis.text.x = element_text(size = 15, angle = 90, hjust = 1, vjust = 0)) +
    ylab("Average rating")
ggtitle("Average Rates of genres")

# Who has rated the most movies?
edx %>% 
    group_by(userId) %>% 
    summarise(avg = mean(rating),
              median = median(rating),
              count = n()) %>% 
    summary()

# Ratings are normally distributed
edx %>% 
    group_by(userId) %>% 
    summarise(mean = mean(rating),
              median = median(rating)) %>% 
    ggplot() +
    geom_density(aes(mean, fill = "mean"), alpha = 0.3) +
    geom_density(aes(median, fill = "median"), alpha = 0.3) +
    xlab("Rating") +
    ylab("Density") +
    ggtitle("Users' average ratings") +
    labs(fill = "")

# Users' activity
edx %>% 
    group_by(userId) %>% 
    summarise(count = n()) %>% 
    ggplot() +
    geom_density(aes(count, fill = "count"), alpha = 0.3) +
    xlab("Count") +
    ylab("Density") +
    ggtitle("Users' activity") +
    labs(fill = "")

# split train and test sets
set.seed(1)
ind <- createDataPartition(y = edx$rating, times = 1, p = 0.3, list = F)
train_set <- edx[-ind, ]
test_set <- edx[-ind, ]

# semi_join to double check that all users and movies in the test set are also included in the train set
test_set <- 
    test_set %>% 
    semi_join(train_set, by = "movieId") %>% 
    semi_join(train_set, by = "userId")

mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(mu_hat, test_set$rating)

# create a data frame to keep the performances for different models
rmse_results <- tibble(
    method = c("Project goal", "Simple average"),
    RMSE = c(0.86490, naive_rmse)
)
mu_hat
rmse_results %>% kable()

mu <- mean(train_set$rating)
movie_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarise(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

predicted_ratings <- mu + test_set %>% 
    left_join(movie_avgs, by = "movieId") %>% 
    pull(b_i)
movie_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, tibble(
    method = "Movie Effect",
    RMSE = movie_rmse)
)
rmse_results %>% kable()

# Calculate user effects
user_avgs <- 
    train_set %>% 
    left_join(movie_avgs, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u = mean(rating - mu - b_i))

# Predict
predicted_ratings <- test_set %>% 
    left_join(movie_avgs, by = "movieId") %>% 
    left_join(user_avgs, by = "userId") %>% 
    mutate(pred = mu + b_i + b_u) %>% 
    pull(pred)

movie_user_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- 
    rmse_results %>% 
    bind_rows(tibble(
        method = "Movie + User Effects",
        RMSE = movie_user_rmse
    ))
rmse_results %>% kable()

# cross-validation

lambdas <- seq(0, 10, 0.5)
lambda_look <- function(l, train_set, test_set) {
    mu <- mean(train_set$rating)
    b_i <- train_set %>% 
        group_by(movieId) %>% 
        summarise(b_i = sum(rating - mu) / (n() + l))
    
    b_u <- train_set %>% 
        left_join(b_i, by = "movieId") %>% 
        group_by(userId) %>% 
        summarise(b_u = sum(rating - mu - b_i) / (n() + l))
    
    predicted_ratings <- 
        test_set %>% 
        left_join(b_i, by = "movieId") %>% 
        left_join(b_u, by = "userId") %>% 
        mutate(pred = mu + b_i + b_u) %>% 
        pull(pred)
    RMSE(predicted_ratings, test_set$rating)
}

rmses <- sapply(lambdas, lambda_look, train_set = train_set, test_set = test_set)
qplot(lambdas, rmses)

# more specified search
lambdas <- seq(0, 1, 0.1)
rmses <- sapply(lambdas, lambda_look, train_set = train_set, test_set = test_set)
qplot(lambdas, rmses)
lambdas[which.min(rmses)]

l <- 0.4
mu <- mean(train_set$rating)
b_i <- train_set %>% 
    group_by(movieId) %>% 
    summarise(b_i = sum(rating - mu) / (n() + l))

b_u <- train_set %>% 
    left_join(b_i, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u = sum(rating - mu - b_i) / (n() + l))

predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    mutate(pred = mu + b_i + b_u) %>% 
    pull(pred)
regularized_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <-
    rmse_results %>% 
    bind_rows(tibble(
        method = "Movie, User Effects, Regularized",
        RMSE = regularized_RMSE
    ))
rmse_results %>% kable()

library(recosystem)
set.seed(123)
train_data <- with(train_set,
                   data_memory(user_index = userId,
                               item_index = movieId,
                               rating = rating))
test_data <- with(test_set,
                  data_memory(user_index = userId,
                              item_index = movieId,
                              rating = rating))

r <- recosystem::Reco()

# we perform without extra tuning process for the sake of simplicity
r$train(train_data, opts = c(niter = 60))
y_hat_reco <- r$predict(test_data, out_memory())
rmse_results <- bind_rows(rmse_results,
                          tibble(
                              method = "Matrix Factorization",
                              RMSE = RMSE(y_hat_reco, test_set$rating)
                          ))
rmse_results %>% kable()

# Since we finished finding our hyperparameter, lambda, we can use the whole edx set from now on.
mu_final <- mean(edx$rating)

# movie effect
b_i_final <- 
    edx %>% 
    group_by(movieId) %>% 
    summarise(b_i_final = sum(rating - mu_final) / (n() + l))

# user effect
b_u_final <- 
    edx %>% 
    left_join(b_i_final, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u_final = sum(rating - mu_final - b_i_final) / (n() + l))

pred_validation <- 
    validation %>% 
    left_join(b_i_final, by = "movieId") %>% 
    left_join(b_u_final, by = "userId") %>% 
    mutate(pred_final = mu_final + b_i_final + b_u_final) %>% 
    pull(pred_final)

RMSE_validation_regularized <- RMSE(pred_validation, validation$rating)
rmse_results <- 
    rmse_results %>% 
    bind_rows(
        tibble(
            method = "Movie, User Effects, Regularized (validation)",
            RMSE = RMSE_validation_regularized
        )
    )
rmse_results %>% kable()

set.seed(123)
train_data <- with(edx,
                   data_memory(user_index = userId,
                               item_index = movieId,
                               rating = rating))
test_data <- with(validation,
                  data_memory(user_index = userId,
                              item_index = movieId,
                              rating = rating))

r <- recosystem::Reco()

# we perform without extra tuning process for the sake of simplicity
r$train(train_data, opts = c(niter = 60))
y_hat_reco <- r$predict(test_data, out_memory())
rmse_results <- bind_rows(rmse_results,
                          tibble(
                              method = "Matrix Factorization (validation)",
                              RMSE = RMSE(y_hat_reco, validation$rating)
                          ))
rmse_results %>% kable()