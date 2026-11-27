# SOCIALPULSE - Social Media Sentiment Analysis



## 1\. Project Overview



SOCIALPULSE is a Big Data Analytics project that analyzes social media tweets and identifies their sentiment as Positive, Negative, Neutral, or Irrelevant.



The project uses Python, PySpark, Machine Learning, and Streamlit to process, analyze, visualize, and predict sentiment from social media data.



## 2\. Objectives



* Analyze large volumes of social media tweets.
* Clean and preprocess raw tweet data.
* Identify Positive, Negative, Neutral, and Irrelevant sentiments.
* Analyze sentiment patterns across different topics.
* Visualize sentiment distribution using interactive charts.
* Predict the sentiment of newly entered tweets.
* Generate downloadable sentiment analysis reports.



## 3\. Technologies Used



* Python
* Pandas
* PySpark
* Scikit-learn
* TF-IDF Vectorization
* Linear Support Vector Machine (Linear SVM)
* Streamlit
* Matplotlib  Streamlit Charts
* Joblib



## 4\. Dataset



The project uses Twitter training and validation datasets.



The cleaned training dataset contains approximately 71,347 tweets.



Sentiment categories:



* Negative
* Positive
* Neutral
* Irrelevant



The dataset also contains different topics related to products, companies, games, and other subjects.



## 5\. Project Structure



SOCIALPULSE

â”‚

â”œâ”€â”€ dashboard

â”‚   â””â”€â”€ app.py

â”‚

â”œâ”€â”€ dataset

â”‚   â”œâ”€â”€ twitter\_training.csv

â”‚   â”œâ”€â”€ twitter\_validation.csv

â”‚   â”œâ”€â”€ twitter\_training\_cleaned.csv

â”‚   â”œâ”€â”€ twitter\_validation\_cleaned.csv

â”‚   â”œâ”€â”€ model\_comparison.csv

â”‚   â””â”€â”€ experiment\_results.csv

â”‚

â”œâ”€â”€ models

â”‚   â”œâ”€â”€ best\_sentiment\_model.pkl

â”‚   â”œâ”€â”€ best\_tfidf\_vectorizer.pkl

â”‚   â”œâ”€â”€ backup\_sentiment\_model.pkl

â”‚   â””â”€â”€ backup\_tfidf\_vectorizer.pkl

â”‚

â”œâ”€â”€ notebooks

â”‚   â”œâ”€â”€ 01\_data\_exploration.py

â”‚   â”œâ”€â”€ 02\_data\_preprocessing.py

â”‚   â”œâ”€â”€ 03\_sentiment\_visualization.py

â”‚   â”œâ”€â”€ 04\_sentiment\_model.py

â”‚   â”œâ”€â”€ 05\_model\_comparison.py

â”‚   â”œâ”€â”€ 06\_save\_best\_model.py

â”‚   â”œâ”€â”€ 07\_test\_model.py

â”‚   â””â”€â”€ 08\_experiment\_models.py

â”‚

â”œâ”€â”€ src

â”‚   â””â”€â”€ spark\_processing.py

â”‚

â”œâ”€â”€ README.md

â””â”€â”€ requirements.txt



## 6\. Methodology



### Step 1: Data Collection

Load the Twitter training and validation datasets.



### Step 2: Data Preprocessing

* Remove missing tweets.
* Remove duplicate records.
* Convert text to lowercase.
* Remove URLs and usernames.
* Remove special characters.
* Remove unnecessary spaces.



### Step 3: Big Data Processing

Use PySpark to process tweet data and perform sentiment and topic-wise analysis.



### Step 4: Data Visualization

Visualize sentiment distribution, trending topics, and topic-wise sentiment patterns.



### Step 5: Model Training

Train machine learning models using TF-IDF features and Linear SVM classification.



### Step 6: Model Evaluation

Compare model performance using accuracy, precision, recall, and F1-score.



### Step 7: Sentiment Prediction

Allow users to enter a tweet and predict its sentiment through the Streamlit dashboard.



## 7\. Model Performance



The final experimental model uses combined Word-Level and Character-Level TF-IDF features with Linear SVM.



Validation results:



| Metric | Score |

|---|---:|

| Accuracy | 96.6% |

| Neutral Precision | 97.2% |

| Neutral Recall | 96.8% |

| Neutral F1-score | 97% |



The model supports four sentiment categories:



* Positive
* Negative
* Neutral
* Irrelevant



## 8\. Dashboard Features



SOCIALPULSE provides:



1. Overall dataset statistics.
2. Sentiment distribution bar chart.
3. Sentiment distribution pie chart.
4. Sentiment comparison across topics.
5. Top 10 trending topics.
6. Topic-wise sentiment heatmap.
7. Topic-wise sentiment analysis.
8. Tweet exploration with sentiment filters.
9. Real-time tweet sentiment prediction.
10. Downloadable sentiment analysis report.



## 9\. Installation and Execution



### Step 1: Clone the repository



git clone YOUR\_GITHUB\_REPOSITORY\_LINK



### Step 2: Open the project folder



cd SOCIALPULSE



### Step 3: Install dependencies



pip install -r requirements.txt



### Step 4: Run the Streamlit dashboard



python -m streamlit run dashboard/app.py



The dashboard will open in your web browser.



## 10\. Future Enhancements



* Improve sentiment prediction for ambiguous tweets.
* Support real-time social media data collection.
* Add multilingual sentiment analysis.
* Integrate advanced deep learning models.
* Provide more detailed sentiment insights and reports.



## 11\. Conclusion



SOCIALPULSE demonstrates how Big Data processing and Machine Learning can be combined to analyze social media sentiment.



The project processes Twitter data, identifies sentiment patterns, visualizes insights, and predicts the sentiment of new tweets through an interactive dashboard.



It helps demonstrate the practical application of Big Data Analytics in understanding social media opinions.

