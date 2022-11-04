# Propensity Model for Book Reading

## Goals:
 * Based on the book ratings given by readers for a period of January through November 2016, predict the book reading propensities in the following month (December 2016).
 * Estimate the confidence for such predictions.
 
## Methodology
### Propensity Model as a Recommender System
 * Considering the nature of the provided data (book ratings offered by readers), this problem lands itself with the recommendation systems.
 * Specifically, a broad class of recommenders known as "collaborative filtering" appears to be the best first step towards predicting book propensities, based entirely on so called "user-item interactions".
### Explicit/Implicit Interactions
 * It should be noted that the provided dataset contains __both implicit and explicit interactions:__
     * __Explicit Interaction:__ A user entered a rating of a certain book
     * __Implicit Interaction:__ All we know is that the user had read the book (but never bothered to share their opinion).
 * Some methods allow including implicit interactions (e.g., SVD++); some were not built for that.
 * For the sake of simplicity, we will henceforth __focus entirely on the explicit interactions.__ In practice, this means that we drop all rows with `rating = 0`.
### Choosing the Method
 * In the field of the collaborative filtering, there are several major strategies. The ones that are considered most competitive are as follows:
     * __Matrix Factorization__-based (MF) algorithms
     * __Deep Neural Network (DNN)__ models
     * __Hybrid__ models
 * In this project, we will focus on the __Matrix Factorization__-based models. Let us briefly list their pros and cons:
     * __Pros:__ 
         * Known to be able to produce quite accurate predictions, through finding the "latent factors"
         * Scalable, during both training and serving
         * Can handle the issue of "folding" (spurious predictions that occur when different clusters happen to be close to each other in the embedding space, purely by chance) that DNN models may suffer from
     * __Cons:__ 
         * Originally MF models could not handle the issue of the "cold start": a situation when a trained model is queried with previously unseen user/item. (Solution: Therre exist heuristic methods, e.g. average embeddings, that help handle this problem)
         * MF models cannot include external features, i.e. anything other than user-item interactions. This imposes a limitation on how much can be learned from the data, and explains why DNN models sometimes outperform the MF-based recommenders

## Main Steps, with corresponding notebooks

### 1. Download the data (`step_1_download_the_data.ipynb`)

As a train/test data, we use the table of reader-book interactions for the genre "Fantasy/Paranormal", from the GoodReads public dataset.

### 2. Filter the records, and save them in a new file (`step_2_filter_the_data.ipynb`)
     * Only the records from the year 2016
     * Only the records where the rating is an integer number greater than zero (`raiting = 0` means no rating was provided)
     
     
### 3. Create the train/test datasets (`step_3_create_train_val_test_sets.ipynb`)
### 4. Select the best model: compare different algorithms, tune hyperparameters, evaluate the performance (`step_4_select_evaluate_model_mf.ipynb`)
### 5. Create, train, and run a propensity model (`step_5_propensity_model.ipynb`)

## References & Citations

 * Mengting Wan, Julian McAuley, ["Item Recommendation on Monotonic Behavior Chains"](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/recsys18_mwan.pdf), in RecSys'18. [bibtex](https://dblp.uni-trier.de/rec/conf/recsys/WanM18.html?view=bibtex)
 
 * Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, ["Fine-Grained Spoiler Detection from Large-Scale Review Corpora"](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/acl19_mwan.pdf), in ACL'19. [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/acl/WanMNM19)
 
 * ["Build a Recommendation Engine With Collaborative Filtering"](https://realpython.com/build-recommendation-engine-collaborative-filtering/) by Abhinav Ajitsaria
 
 * ["Collaborative Filtering with Machine Learning and Python"](https://rubikscode.net/2020/04/27/collaborative-filtering-with-machine-learning-and-python/) by Nikola M. Zivkovic
 
 * ["Machine Learning: Recommendation Systems"](https://developers.google.com/machine-learning/recommendation) (A course on Google Developer, with examples that focus on the use of the Tensorflow)
 
 * ["Collaborative Filtering on Ordinal User Feedback"](https://www.ijcai.org/Proceedings/13/Papers/449.pdf) by Y. Koren and J. Sill. Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence, 2011
