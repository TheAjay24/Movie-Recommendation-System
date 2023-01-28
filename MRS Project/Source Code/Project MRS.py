#!/usr/bin/env python
# coding: utf-8

# # MOVIE RECOMMENDATION SYSTEM

# #Importing the libraries

# In[21]:


import numpy as np
import pandas as pd 
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# #Data Collection and Preprocessing

# In[22]:


#Loading the data from the csv file to a pandas dataframe

movies_data = pd.read_csv(r"C:\Users\AJAY\Desktop\Data Science\Project\Data Set\movies.csv")


# In[23]:


#Printing the first five rows of the dataframe

movies_data.head()


# In[24]:


#Number of Rows and Columns in the dataframe

movies_data.shape


# In[25]:


#Selecting the relevant feature for recommendation / feature selection

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[26]:


#Replacing the missing values/Null values with Null string 

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# In[27]:


#Combining all the five selected features together

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[28]:


print(combined_features)


# In[29]:


#Convert the text data into feature vectors

vectorizer = TfidfVectorizer()


# In[30]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[31]:


print(feature_vectors)


# # Cosine Similarity

# In[34]:


#Getting the similarity scores using the cosine similarity

similarity = cosine_similarity(feature_vectors)


# In[35]:


print(similarity)


# In[37]:


print(similarity.shape)


# In[38]:


#Getting the movie name from the user 

movie_name = input("Enter your favourite movie: ")


# In[40]:


#Creating a list with all the movie name given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[41]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[42]:


close_match = find_close_match[0]
print(close_match)


# In[44]:


#finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title==close_match]['index'].values[0]
print(index_of_the_movie)


# In[46]:


#Getting a list of similar movie

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[47]:


len(similarity)


# In[48]:


#Sorting the movies based on the similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)


# In[55]:


#Print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<=30):
    print(i, '.',title_from_index)
    i+=1


# # Hey there! I am a Movie Recommender!  

# In[74]:


movie_name = input('Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<=30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:




