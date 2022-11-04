# Propensity Model for Book Reading

## Goals:
 * Based on the book ratings given by readers January through November 2016, predict the propensities for December 2016.
 * Also, estimate the confidence for such predictions
 
## Methodology
 * Considering the nature of the provided data (book ratings offered by readers), this problem lands itself with the recommendation systems.
 * Specifically, a broad class of recommenders known as "collaborative filtering" appears to be the best first step towards predicting book propensities, based entirely on so called "user-item interactions".
 * It should be noted that the provided dataset contains __both implicit and explicit interactions:__
     * __Explicit Interaction:__ A user entered an actual rating of a certain book
     * __Implicit Interaction:__ A user had read the book, but never bothered to rate it
 * Some methods allow including implicit interactions (e.g., SVD++); some were not built for that.
 * For the sake of simplicity, we will henceforth __focus entirely on the explicit interactions.__ In practice, this means that we drop all rows with `rating = 0`.
 * In the field of the collaborative filtering, there are 

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
