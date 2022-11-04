# Propensity Model for Book Reading

#### Author: Kostyantyn Borysenko

## Goals:
 * Based on the book ratings given by readers for a period of January through November 2016, predict the book reading propensities in the following month (December 2016).
 * Estimate the confidence for such predictions.
 
## Methodology
### Propensity Model as a Recommender System
 * Considering the nature of the provided data (book ratings offered by readers), this problem lands itself with the recommendation systems.
 * Specifically, a broad class of recommenders known as "collaborative filtering" appears to be the best first step towards predicting book propensities, based entirely on so called "user-item interactions".
### Input Data and Explicit/Implicit Interactions
 * It should be noted that the provided dataset contains __both implicit and explicit interactions:__
     * __Explicit Interaction:__ A user entered a rating of a certain book
     * __Implicit Interaction:__ All we know is that the user had read the book (but never bothered to share their opinion).
 * Some methods allow including implicit interactions (e.g., SVD++); some were not built for that.
 * For the sake of simplicity, we will henceforth __focus entirely on the explicit interactions.__ In practice, this means that we drop all rows with `rating = 0`.
### Choosing the Propensity Model Type
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
         
### Notes on the Model Evaluation & Comparison

 * Through a series of experiments, we have found that the SVD (Singular Value Decomposition) algorithm proved to be marginally better than the NMF (Non-negative Matrix Factorization), when evaluated with several accuracy metrics: RMSE (root mean square error), MAE (mean absolute error), and FCP (Fraction of Concordant Pairs).
 * SVD also proved to be much faster to train in comparison to the NMF, while using the same number of epochs.
 * It is worth noting that the "Baseline Only" algorithm (i.e., an algorithm predicting the baseline estimate for given user and item: global mean, adjusted with user/item biases) was not too far behind. This suggests that perhaps our "perfect model" would benefit from a DNN-based, or a hybrid approach, when the ranks that capture the user-item interaction, are accompanied by additional, "external" features (e.g., complementary data about a user, or an item).
         
### Estimating the Confidence for the Propensity Model's Predictions
  * Our trained recommender can be easily converted into a propensity model, which is in essense, a binary classifier: it would ultimately generate a "yes or no" answer when asked if a reader X would like a book Y.
 * We define the confidence for the results of such propensity model as a probability that the model will make a correct prediction for the selected user-item pair.
 * As a first approximation, we can estimate this probability globally, for the entire dataset. 
 * If the validation set size is $n$ and the number of correct predictions is $n^c$, then a point estimate of the the confidence $C$ is $$\hat{C} = \frac{n^c}{n}$$
 * We can estimate the value of the confidence on the train set, using the model we trained, which is a common practice for recommender systems. In order to minimize a bias of such estimate, we can use a K-fold cross-validation procedure. 
 * We could potentially achieve a higher accuracy for the confidence estimates, by calculating the confidence to every reader individually:  $$\hat{C_{u}} = \frac{n^c_{u}}{n_{u}},$$ where $n_{u}$ is a total number of ratings left by the user, and $n^c_{u}$ is the number of the user propensities that our model guessed correctly. This of course, can only be applied to a known reader. For the case of a "cold start", we have to rely on the global estimate $C$.
 * In the current implementation, the confidence is estimated as a weighted average of the global and user-based confidence levels: $$\hat{C_{u}} = \frac{n_{min} C + n^c_{u}}{n_{min} + n_{u}},$$ where $n_{min}$ is the minimal necessary number of ratings.

## Test Results

 * When evaluated on the test set, our best trained SVD recommendation model (128 latent factors, 100 epochs, with some regularization) demonstrates a low bias:
     * Avgerage Rating:           4.03
     * Avgerage Predicted Rating: 4.02
 * On the other hand, the __estimated accuracy of 65.85%__ (lower bound estimate) suggests that our model suffers from a considerable variance. 
 * With `MAE = 0.6681`, one can see that a substantial number of user-item recommendations get misclassified near the decision boundary ("no" corresponds to `rank < 4`; "yes" is for `rank >= 4`).
 * The model's performance could be further improved with the use of one of data resampling techniques. This approach has a potential of lowering the variance, thereby boosting the overall accuracy.

## Conclusions

 * A matrix factorization-based recommendation model (SVD) trained on the "GoodReads" explicit book rating dataset showed the following performance:
     * RMSE: 0.8599
     * MAE:  0.6681
     * FCP:  0.5855
 * It proved to have a low bias but suffered from a considerable variance. 
 * A binary classifier (propensity model) built based on the aforementioned recommendation system demostrated the accuracy of 65.85% on the test set. Further improvement could potentially be achieved through data resampling techniques.
 * Use of a DNN-based, or a hybrid model could lead to a better accuracy, especially if external features are incorporated.

## Appendix: A Few Notes on the Main Steps

### 1. Download the data (`step_1_download_the_data.ipynb`)

 * As a train/test data, we use the table of reader-book interactions for the genre "Fantasy/Paranormal", from the GoodReads public dataset.

### 2. Filter the records, and save them in a new file (`step_2_filter_the_data.ipynb`)
 * Only the records from the year 2016
 * Only the records where the rating is an integer number greater than zero (`raiting = 0` means no rating was provided)
    
### 3. Create the train/test datasets (`step_3_create_train_val_test_sets.ipynb`)
 * Train Set: Book ratings for January - Novemver 2016
 * Test Set: Book ratings for December 2016
### 4. Select the best model: compare different algorithms, tune hyperparameters, evaluate the performance (`step_4_select_evaluate_model_mf.ipynb`)
 * Focus on evaluating two MF-based model types:
     * SVD
     * NMF
 * Note: We do not consider the SVD++ model here, as it was designed to handle the implicit interactions, which we excluded from this study.
### 5. Create, train, and run a propensity model (`step_5_propensity_model.ipynb`)
 * Create a `Propensity` class
 * Train and evaluate a selected model
 * Serve the results
 
## References & Citations

 * Mengting Wan, Julian McAuley, ["Item Recommendation on Monotonic Behavior Chains"](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/recsys18_mwan.pdf), in RecSys'18. [bibtex](https://dblp.uni-trier.de/rec/conf/recsys/WanM18.html?view=bibtex)
 
 * Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, ["Fine-Grained Spoiler Detection from Large-Scale Review Corpora"](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/acl19_mwan.pdf), in ACL'19. [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/acl/WanMNM19)
 
 * ["Build a Recommendation Engine With Collaborative Filtering"](https://realpython.com/build-recommendation-engine-collaborative-filtering/) by Abhinav Ajitsaria
 
 * ["Collaborative Filtering with Machine Learning and Python"](https://rubikscode.net/2020/04/27/collaborative-filtering-with-machine-learning-and-python/) by Nikola M. Zivkovic
 
 * ["Machine Learning: Recommendation Systems"](https://developers.google.com/machine-learning/recommendation) (A course on Google Developer, with examples that focus on the use of the Tensorflow)
 
 * ["Collaborative Filtering on Ordinal User Feedback"](https://www.ijcai.org/Proceedings/13/Papers/449.pdf) by Y. Koren and J. Sill. Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence, 2011
 
 * [Scikit Suprise Library](https://surprise.readthedocs.io/en/stable/index.html)
