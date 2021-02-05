# Social Media Sentiment Analysis Engine
Download the command line application here.

### Navigate this Repository

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
