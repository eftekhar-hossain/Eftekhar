
## [Project 1: Polarity Detection of Bengali Book Reviews Using Machine Learning](https://github.com/eftekhar13/Bengali-Book-Reviews)
- Created a tool that can detect the sentiment polarity (either **positive or negative**) of Book reviews written in Bengali Text. 
- Collected **1k** book reviews from different online book shops as well as social media groups. Among these reviews **528** reviews are labelled as positve and **472** reviews are labelled as negative sentiment.
- Extract **Unigram, Bigram and Trigram** features from the cleaned Text and use the **TF-idf vectorizer** as a feature extraction technique.
- Employed different machine learning classifiers for the classification purpose. The used classifiers are **Logistic Regression, Decision Tree, Multinomial Naive Bayes, Support Vector Machine** and so on.
- Evaluate the performance of the classification for every gram feature. **Accuracy, Precision, Recall, F1-score, ROC curve and Precision-Recall curve** used as evaluation metrics.
- Finally, created a client facing API using Flask. [App link](https://sa-book-review.herokuapp.com/)
- Publication: [Link](https://www.researchgate.net/publication/342673109_Sentiment_Polarity_Detection_on_Bengali_Book_Reviews_Using_Multinomial_Naive_Bayes)

![book](/images/book.png)


## [Project 2: Bangla News Headlines Categorization Using Gated Recurrent Unit (GRU)](https://github.com/eftekhar13/Bangla-News-Headlines-Categorization)
- Created a tool that can categorizes the Bengali news headlines into six category (**National, Politics, International, Sports, Amusement, IT**) using deep recurrent neural network.
- A dataset of **0.13 Million** news headlines is created. Chrome web scrapper used for scraping the news headlines from different Bengali online news portals such as **Dainik Jugantor, Dainik Ittefaq, Dainik Kaler Kontho** and so on.    
- **Word embeeding** feature represtations technique is used for extracting the semantic meaning of the words.
- A deep learning model has been built by using a **bidirectional gated recurrent network**.
- Finally, the model performance is evaluated using various evaluation measures such as **confusion matrix, accuracy , precision, recall and f1-score**.  

![headline](/images/headline.png)


## [Project 3: Sentiment Analysis of Bengali Restaurant Reviews Using Machine Learning](https://github.com/eftekhar13/Bengali-Restaurant-Reviews)
- Created a tool that can identify the sentiment of a restaurant review written in Bengali Text. It classifies a review as **positive or negative** sentiment.   
- Collected **1.4k** Bengali restaurant reviews from different social media groups of food or restaurant reviews. Among these reviews 630 reviews are labelled as positve and 790 reviews are labelled as negative sentiment.
- Extract **Unigram, Bigram and Trigram** features from the cleaned Text and use the **TF-idf vectorizer** as a feature extraction technique.
- Employed different machine learning classifiers for the classification purpose. The used classifiers are **Logistic Regression, Decision Tree, Multinomial Naive Bayes, Support Vector Machine, Stochastic Gradient Descent** and so on.
- Evaluate the performance of the classification for every gram feature. **Accuracy, Precision, Recall, F1-score, ROC curve and Precision-Recall curve** used as evaluation metrics.
- Finally, created a client facing API using Flask and deployed into cloud using **Heroku**. [App Link](https://sa-restaurant-reviews.herokuapp.com/)
- Publication: [Link](https://ieeexplore.ieee.org/abstract/document/8934655)

![rest](/images/restaurant1.png)


## [Project 4: Sentiment Analysis of Bangla News Comments Using Machine Learning](https://github.com/eftekhar13/Bangla-News-Comments)
- Developed a machine learning model that can classify the sentimental category (**positive, negative and neutral**) of a news comment written in Bangla Text.
- For the implementation a publicly available [dataset](https://data.mendeley.com/datasets/n53xt69gnf/3) of **12k** news comments have been used. 
- To create the system TF-idf feature extraction technique with n-gram features have been used.
- Analysed the performance of different machine learning algorithms for n-gram feature by using various evaluation metrics such as **accuracy, precision, recall and f1-score**.

![comment](/images/comment.png)


## [Project 5: Bengali Document Categorization Using ConV-LSTM Net](https://github.com/eftekhar13/Bengali-Document-Categorization)

- Created a tool that can categorizes the Bengali news articles into 12 diffferent categories (**Art, Politics, International, Sports, Science, Economics, Crime, Accident, Education, Entertainment, Environment, Opinion**) using **Deep Learning**.
- A publicly available [dataset](https://data.mendeley.com/datasets/xp92jxr8wn/2) of **0.1 Million** news articles is used to develop the system. The dataset consist 12 different categories news articles.      
- **Word embeeding** feature represtations technique is used for extracting the semantic meaning of the words.
- A deep learning model has been built by using a **Convolutional Neural Network and Long Short Term Memory**.
- The model performance is evaluated using various evaluation measures such as **confusion matrix, accuracy , precision, recall and f1-score**.
- Finally, developed a client facing API using **flask** and **heroku**.
- Here is the developed Flask App : [Document Categorizer App](https://bangla-document-categorization.herokuapp.com/)

![document](/images/document.png)

## [Project 6: Word Embedding on a Bengali Text Corpus](https://github.com/eftekhar13/Word-Embedding-on-Bangla-Text)
- Created a word embedding model for Bangla text corpus.
- Used Word2Vec algorithm.
- Used a publicly availabe [dataset](https://data.mendeley.com/datasets/xp92jxr8wn/2) of **0.1 Milion** Bangla news articles.
- Visualized the word similarity using **t-sne** plot. 

![wordvec](/images/word2vec.png)



