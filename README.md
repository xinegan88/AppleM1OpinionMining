# Social Media Sentiment Analysis Engine

### Navigate this Repository

![github_directory](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/github_directory.png)

## What do users think about the Apple M1 chip?

In December 2020, Apple launched three products, Mac Mini, MacBook Air, and MacBook Pro featuring the M1 chip. This was a departure from previous iterations of these products which used Intel chips. Apple claimed that the new chip would offer improved performance and efficiency at a better price point.

In order to determine if users felt that the M1 chip was living up to Apple’s claims, I designed a sentiment analysis engine to extract data from Tweets and Reddit posts/comments and analyze the user sentiments. Then, I built a model to predict if a given text blurb from a user was positive, negative, or neutral so that I could make generalizations about   the user experience in each category.

### Methodology
	* Build an application to collect data.
	* Preprocess: Clean text, remove stops and lemmatize.
	* Extract features: Text length, POS tags, subjectivity, and compound polarity.
	* Label data as positive, negative, or neutral based on compound polarity.
	* Create a model that will predict if an observation is positive, negative, or neutral.
	* Make generalizations regarding each category.

### Data Pipeline
In order to collect the data, I ran my script at various times each day, and saved each data pull from the API with a time stamp, then periodically pulled batches of the raw data.

![DataPipeline](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/data_pipeline.png)     

### Exploratory Data Analysis
#### Positive Reception
![DataClasses](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/class_frequencies.png)     

 Positive Class (1.0): 6322
 Neutral Class (0.0): 5696
 Negative Class (-1.0): 1628          
 
 It is a little suspicious that the neutral class is so large. However, out of all three classes, the neutral class had the most instances of foreign words. It could be that the foreign words are causing a mislabeling. This could mean that the class imbalance may be more or less pronounced, depending on the sentiment of the foriegn words.   

#### Negative Opinions
Among the negative comments we observe frequent instances of the words "air", "pro", "iphone", and "ipad". This suggests that there is some link between these products and user dissatisfaction. Since one of Apple's main claims was that the M1 would foster compatibility among other products in the Apple Universe, including allowing iPhone and iPad apps to run natively on the machine, it would be worth further investigation to determine if users think that the M1 is living up to these claims.     
![NegativeWords](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/neg_words.png)

### Models
#### 1. Niave Bayes
I used GaussianNB from sklearn to to make predictions about polarity. Since there was a class imbalance, I ran the model with and without SMOTE, and with cross validation and chose from the best scores.      
![NB](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/NB_SMOTE_TABLE.png)    

![NB](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/NB_SMOTE_ROC.png)    


#### 2. Support Vector Classifier
I also used sklearn.SVM SVC to see if I could improve my results. I fit the model with hyper parameters using GridSearchCV. The performance of the model was impressive, however I would like to further investigate if overfitting is occuring.    
![SVC](https://github.com/xinegan88/AppleM1OpinionMining/blob/main/images/SVC_TABLE_ROC.png)    

### Recommendations
* Collect more data regarding the customer's perception of compatibility among other products in the Apple universe. Work towards increased compatibility in later updates and releases.
* Develop methods to deal with foriegn words when collecting social media data.
* Capitialize on opportunities to highlight performance by reaching out to users with positive experiences and asking them about specific performance benchmarks. Expand efforts to learn more about user perception of performance.


### Future Work
* Expand application to access data from additional platforms.
* Take advantage of time data to provide insights into user opinions over time.
* Stream data to a dashboard to analyze and update changing opinions in realtime.
