#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import make_pipeline
from PIL import Image
from wordcloud import STOPWORDS, WordCloud
import requests
from bs4 import BeautifulSoup
import pickle
import warnings
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Ignore warning messages
warnings.filterwarnings('ignore')


# In[2]:


phish_data = pd.read_csv("phishing_site_urls.csv", encoding='utf-8')       # Load the phishing dataset


# In[3]:


phish_data.head()                                                          # Display the first few rows of the dataset


# In[4]:


phish_data.tail()                                                         # Display the last few rows of the dataset


# In[5]:


phish_data.info()                        # Display dataset information


# In[6]:


phish_data['Label'].value_counts()           # Count the number of samples in each class


# In[7]:


phish_data.isnull().sum()                  # Check for missing values in the dataset


# In[8]:


melted = pd.melt(phish_data, value_vars=["Label"])


# In[9]:


sns.countplot(x='value', data=melted)           # Create a count plot to visualize class distribution


# In[10]:


# Initialize tokenizer and stemmer for text preprocessing
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")


# In[11]:


# Tokenize, stem, and join the text data
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t))
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))


# In[12]:


# Data Preprocessing: Convert text to lowercase and remove stopwords
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

phish_data['text_sent'] = phish_data['text_sent'].apply(lambda x: ' '.join(word.lower() for word in x.split() if word.lower() not in stop_words))


# In[13]:


phish_data.sample(5)                   # Displaying a sample of preprocessed data


# In[14]:


# Slice the dataset into "bad" and "good" sites
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']


# In[15]:


bad_sites.head()


# In[16]:


good_sites.head()


# In[17]:


def scrape_website_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.text for p in soup.find_all('p')])  # Extract text from <p> tags

        # Data Preprocessing: Convert text to lowercase and remove stopwords
        text = ' '.join(word.lower() for word in text.split() if word.lower() not in stop_words)

        return text
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""


# In[18]:


# List of URLs to scrape text from
scrape_urls = ['https://uims.cuchd.in/uims/', 'https://www.wikipedia.org']


# In[19]:


# Combine and scrape text from the specified URLs
website_text = " ".join([scrape_website_text(url) for url in scrape_urls])


# In[20]:


# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_feature = tfidf_vectorizer.fit_transform([website_text])


# In[21]:


# Function to create and plot a word cloud from text data

def plot_wordcloud_from_data(data, title):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com', 'http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=120,
                          random_state=42)
    wordcloud.generate(data)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title, fontdict={'size': 40, 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()


# In[22]:


# Generate and plot a word cloud for good URLs
good_sites_text = " ".join(phish_data[phish_data['Label'] == 'good']['text_sent'])
plot_wordcloud_from_data(good_sites_text, title='Word Cloud for Good URLs')


# In[23]:


# Generate and plot a word cloud for bad URLs
bad_sites_text = " ".join(phish_data[phish_data['Label'] == 'bad']['text_sent'])
plot_wordcloud_from_data(bad_sites_text, title='Word Cloud for Bad URLs')


# In[24]:


from selenium import webdriver

def scrape_website_text(url):
    try:
        options = webdriver.ChromeOptions()
        # Specify the path to your Chrome browser executable
        options.binary_location = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)  # Omit executable_path
        driver.get(url)
        time.sleep(3)  # Allow time for the page to load
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")
        text = " ".join([p.text for p in soup.find_all('p')])  # Extract text from <p> tags
        return text
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""


# In[25]:


# List of URLs to scrape text from using Selenium
scrape_urls = ['https://www.segurossura.com.co/', 'https://en.wikipedia.org/wiki/Cricket_World_Cup']


# In[26]:


# Combine and scrape text from the specified URLs using Selenium
website_text = " ".join([scrape_website_text(url) for url in scrape_urls])


# In[27]:


# Function to create and plot a word cloud from text data
def plot_wordcloud(text, max_words=400, max_font_size=120, title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42)
    wordcloud.generate(text)

    plt.figure(figsize=(12, 8))
    if image_color:
        image_colors = ImageColorGenerator(np.array(Image.open("star.png")))
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    else:
        plt.imshow(wordcloud, interpolation="bilinear")

    plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()

plot_wordcloud(website_text, title='Word Cloud for Scraped Websites')

plt.show()


# In[28]:


# Handling Imbalanced Data: Resample the minority class (phishing)
phish_data_majority = phish_data[phish_data['Label'] == 'good']
phish_data_minority = phish_data[phish_data['Label'] == 'bad']


# In[29]:


from sklearn.utils import resample

phish_data_minority_upsampled = resample(phish_data_minority,
                                         replace=True,
                                         n_samples=len(phish_data_majority),
                                         random_state=42)
phish_data_upsampled = pd.concat([phish_data_majority, phish_data_minority_upsampled])


# In[30]:


# Create a CountVectorizer object for text feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
cv = CountVectorizer()


# In[31]:


all_text = phish_data.text_sent
# Fit the CountVectorizer on the combined data
cv.fit(all_text)
feature = cv.transform(all_text)  # Transform your text data into features
print(feature[:5].toarray())  # Print the first 5 samples as an example


# In[32]:


trainX, testX, trainY, testY = train_test_split(phish_data_upsampled['text_sent'], phish_data_upsampled['Label'])


# In[33]:


# Initialize a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed


# In[34]:


# Fit and transform the training text data
trainX_tfidf = tfidf_vectorizer.fit_transform(trainX)


# In[35]:


# Transform the test text data using the same vectorizer
testX_tfidf = tfidf_vectorizer.transform(testX)


# In[36]:


# Initialize a Logistic Regression model
lr = LogisticRegression()


# In[37]:


# Hyperparameter Tuning: Grid Search for Logistic Regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(trainX_tfidf, trainY)


# In[38]:


# Get the best hyperparameter
best_C = grid_search.best_params_['C']


# In[39]:


# Train Logistic Regression with the best hyperparameter
lr = LogisticRegression(C=best_C)
lr.fit(trainX_tfidf, trainY)


# In[40]:


# Predict on the test set
predictions = lr.predict(testX_tfidf)


# In[41]:


# Evaluate the model
report = classification_report(testY, predictions)
print(report)


# In[42]:


from sklearn.metrics import accuracy_score
Scores_ml = {}

# Predict on the test set using the trained logistic regression model
lr_predictions = lr.predict(testX_tfidf)

# Calculate accuracy on the test set
lr_accuracy = accuracy_score(testY, lr_predictions)

# Store the accuracy score in the Scores_ml dictionary
Scores_ml['Logistic Regression'] = np.round(lr_accuracy, 2)

# Print the accuracy
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")


# In[43]:


# Display accuracy and classification report for Logistic Regression
#print("Logistic Regression Accuracy:", lr_accuracy)


# In[44]:


print("Logistic Regression Classification Report:")
print(classification_report(testY, lr_predictions, target_names=['Bad', 'Good']))


# In[45]:


lr_confusion_matrix = confusion_matrix(testY, lr_predictions)
lr_confusion_df = pd.DataFrame(lr_confusion_matrix, columns=['Predicted:Bad', 'Predicted:Good'], index=['Actual:Bad', 'Actual:Good'])
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(lr_confusion_df, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[46]:


# Multinomial Naive Bayes
mnb = MultinomialNB()
# Fit the classifier on the TF-IDF transformed training data
mnb.fit(trainX_tfidf, trainY)

# Predict on the test set
mnb_predictions = mnb.predict(testX_tfidf)

# Calculate accuracy on the test set
mnb_accuracy = accuracy_score(testY, mnb_predictions)

Scores_ml['MultinomialNB'] = np.round(mnb_accuracy,2)


# In[47]:


# Predict on the test set using the trained Multinomial Naive Bayes model
mnb_predictions = mnb.predict(testX_tfidf)

# Calculate accuracy on the test set
mnb_accuracy = accuracy_score(testY, mnb_predictions)
mnb_confusion_matrix = confusion_matrix(testY, mnb_predictions)


# In[48]:


# Display accuracy and classification report for Multinomial Naive Bayes
print("\nMultinomial Naive Bayes Accuracy:", mnb_accuracy)
print("Multinomial Naive Bayes Classification Report:")
print(classification_report(testY, mnb_predictions, target_names=['Bad', 'Good']))


# In[49]:


# Confusion Matrix for Multinomial Naive Bayes
mnb_confusion_matrix = confusion_matrix(testY, mnb_predictions)
mnb_confusion_df = pd.DataFrame(mnb_confusion_matrix, columns=['Predicted:Bad', 'Predicted:Good'], index=['Actual:Bad', 'Actual:Good'])
plt.figure(figsize=(8, 6))
sns.heatmap(mnb_confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Multinomial Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[50]:


acc = pd.DataFrame.from_dict(Scores_ml, orient='index', columns=['Accuracy'])


# In[51]:


# Create a bar plot with seaborn
sns.set_style('darkgrid')
sns.barplot(x=acc.index, y=acc['Accuracy'])  # Specify x and y explicitly

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=90)

# Display the plot
plt.show()
acc = pd.DataFrame.from_dict(Scores_ml, orient='index', columns=['Accuracy'])


# In[52]:


from sklearn.pipeline import Pipeline
# Create a pipeline that includes CountVectorizer and Logistic Regression
lr_pipeline = make_pipeline(CountVectorizer(tokenizer=RegexpTokenizer(r'[A-Za-z]+').tokenize, stop_words='english'), LogisticRegression())


# In[53]:


# Split the dataset into training and testing sets
trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)


# In[54]:


# Fit the pipeline on the training data
lr_pipeline.fit(trainX,trainY)


# In[55]:


# Predict using the pipeline
lr_pipeline_predictions = lr_pipeline.predict(testX)
lr_pipeline_accuracy = accuracy_score(testY, lr_pipeline_predictions)
lr_pipeline.score(testX,testY)


# In[56]:


# Display accuracy Logistic Regression from the pipeline
print("Logistic Regression (Pipeline) Accuracy:", lr_pipeline_accuracy)


# In[57]:


# Print the classification report for Logistic Regression from the pipeline
print("Logistic Regression pipeline Classification Report:")
print(classification_report(testY, lr_pipeline_predictions, target_names=['Bad', 'Good']))


# In[58]:


# Confusion Matrix for Logistic Regression from the pipeline
lr_pipeline_confusion_matrix = confusion_matrix(testY, lr_pipeline_predictions)
lr_pipeline_confusion_df = pd.DataFrame(lr_pipeline_confusion_matrix, columns=['Predicted:Bad', 'Predicted:Good'], index=['Actual:Bad', 'Actual:Good'])
plt.figure(figsize=(8, 6))
sns.heatmap(lr_pipeline_confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Logistic Regression (Pipeline) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[59]:


import pickle

# Save the Logistic Regression pipeline model to a file

with open('phishing_lr_pipeline.pkl', 'wb') as file:
    pickle.dump(lr_pipeline, file)

# Load the saved Logistic Regression pipeline model from a file
with open('phishing_lr_pipeline.pkl', 'rb') as file:
    loaded_lr_pipeline = pickle.load(file)

# List of URLs to predict
urls_good = ['https://www.youtube.com/watch?v=h-gTSQ9LXes', 'https://en.wikipedia.org/wiki/Sachin_Tendulkar', 'indianarmy.nic.in', 'https://en.wikipedia.org/wiki/Virat_Kohli', 'https://www.youtube.com/user/chandigarhuniversity']
urls_bad = ['https://10982nklag.com','https://ff-reward-redemption.blogspot.com/','https://djinjrgt.taplink.ws/', 'https://ptonlower20.sbs', 'http://kxtqmjfcqp.duckdns.org']
all_urls = urls_good + urls_bad

# Make predictions using the loaded Logistic Regression pipeline model
results = loaded_lr_pipeline.predict(all_urls)

# Print the results for each URL
for url, result in zip(all_urls, results):
    if result == 'good':
        print(f"URL: {url} -> Predicted: Good")
    else:
        print(f"URL: {url} -> Predicted: Bad")

