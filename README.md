# Machine Learning Project 2 :Machine Learning for Science

`Teammates`: Yan Fu, Shengzhao Xia, Runzhe Liu

For this project we joined the lab in Chair of Economics and Management of Innovation. 

In this project, we did:

1. Data collection: web scrap from Web of Science and get over 350,000 paper records;
2. Name disambiguation: Disambiguation of different authors using same name;
3. Classification: Using models (eg. logistic regression, neural network), and converting keywords as vectors using word2vec to predict if one researcher is dismissed or not.

To reproduce our result one will need to:

1. Install libraries like `gensim`,`sklearn` on their computers;
2. Download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) to do word2vec conversion, we did not include it in our submission because of its large size.
3. Download [wiki-news-300d-1M-subword.vec](https://fasttext.cc/docs/en/english-vectors.html) to do word2vec conversion, we did not include it in our submission because of its large size.


Below we will introduce files and functions in our repository.
And because we cannot upload the whole dataset(the list is sensitive to some extent; and the whole dataset is very large) we comment out the code where origin data is used and save and load the output of preprocessing part. We would include comments because we want to use them to show what we do in pre-processing).

## Auxiliary modules

* ### The folder `DataCollection`
This folder saves the codes used for webscraping and basic data cleaning, and it is *NOT* reproducible because we delete the username and password needed for apply request from WOS. 

* ### `datapreprocessing`

Undis_List_Generation_for_FastText.ipynb is used to create a list of undismissed researchers from database. But due to privacy Data_Complete_clean.txt is missing to run it.

## Working notebook
* ### `Disambiguation.ipynb`: 
The name disambiguation part. We use kNN+LOF outlier dectection in it.

* ### `classification.ipynb`




* ### The folder `Data`

'author_explict_dict.pickle' and not_dis_author_dict_tmp.pickle contain keywords from each dismissed author and corresponding vectors from FASTTEXT.

not_dis_author_dict.pickle and not_dis_author_dict_tmp.pickle contains keywords from each undismissed author in "Set 1" and corresponding vectors from FASTTEXT.

total_no_dis_pub_2.csv and no_dis_auth_vec_2.csv contain publications from undismissed author and vectors from FASTTEXT in "Set 2"

X_set1_fasttext.pickle,y_set1_fasttext.pickle,X_set2_fasttext.pickle,y_set2_fasttext.pickle contain X and y to train our model in the FASTTEXT part.
