# Eftekhar-Projects
This page contains some of my machine learning projects in NLP application.

# [Project 1: Polarity Detection of Bengali Book Reviews Using Machine Learning: Project Overview](https://github.com/eftekhar13/Bengali-Book-Reviews)
- Created a tool that can detect the sentiment polarity (either **positive or negative**) of Book reviews written in Bengali Text. 
- Collected `1k` book reviews from different online book shops as well as social media groups. Among these reviews `528` reviews are labelled as positve and `472` reviews are labelled as negative sentiment.
- Use a custom *stopword list* for removing some words that have not much impact on classification.
- Extract `Unigram, Bigram and Trigram` features from the cleaned Text and use the `TF-idf vectorizer` as a feature extraction technique.
- Employed different machine learning classifiers for the classification purpose. The used classifiers are `Logistic Regression, Decision Tree, Multinomial Naive Bayes, Support Vector Machine` and so on.
- Evaluate the performance of the classification for every gram feature. `Accuracy, Precision, Recall, F1-score, ROC curve and Precision-Recall curve` used as evaluation metrics.
- Finally created a `client facing API` using Flask. 



# [Project 2: Sentiment Analysis of Bengali Restaurant Reviews Using Machine Learning: Project overview](https://github.com/eftekhar13/Bengali-Restaurant-Reviews)
- Created a tool that can identify the sentiment of a restaurant review written in Bengali Text. It classifies a review as **positive or negative** sentiment.   
- Collected **`1.4k`** Bengali restaurant reviews from different social media groups of food or restaurant reviews. Among these reviews `630` reviews are labelled as positve and `790` reviews are labelled as negative sentiment.
- Use Bengali *stopword list* for removing some words that have not much impact on classification.
- Extract `Unigram, Bigram and Trigram` features from the cleaned Text and use the `TF-idf vectorizer` as a feature extraction technique.
- Employed different machine learning classifiers for the classification purpose. The used classifiers are `Logistic Regression, Decision Tree, Multinomial Naive Bayes, Support Vector Machine`, `Stochastic Gradient Descent` and so on.
- Evaluate the performance of the classification for every gram feature. `Accuracy, Precision, Recall, F1-score, ROC curve and Precision-Recall curve` used as evaluation metrics.
- Finally created a `client facing API` using Flask and deployed into cloud using `Heroku`. 
