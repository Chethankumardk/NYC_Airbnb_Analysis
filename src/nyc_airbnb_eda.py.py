#!/usr/bin/env python
# coding: utf-8

# ## Project Title:
# ### "Enhancing Insights from New York City Airbnb Data: A Comprehensive Data Cleaning and Analysis"

# ### Project Overview:
# The project aims to leverage the New York City Airbnb dataset to enhance data quality through cleaning procedures and extract valuable insights through exploratory data analysis (EDA). By addressing data inconsistencies, handling missing values, and exploring patterns in the dataset, the project seeks to provide a robust foundation for subsequent analyses and decision-making.
# 
# #### **Data: https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata **
# 
# ### Background:
# Airbnb has transformed the hospitality industry, providing a unique platform for travelers and hosts to engage in personalized lodging experiences. The New York City Airbnb dataset, as part of Airbnb Inside, captures listing activities, reviews, and calendar details. The dataset offers an opportunity to gain valuable insights into host behaviors, areas of popularity, and pricing dynamics.
# 
# ### Problem Statement:
# The dataset exhibits inconsistencies, missing values, and potential data quality issues that need to be addressed. Cleaning and preparing the data are crucial steps to ensure accurate analyses and reliable insights. Additionally, there is a need to uncover patterns and trends within the dataset that can provide actionable information for hosts and the Airbnb platform.
# 
# ### Objectives:
# 1. **Data Cleaning:**
#    - Identify and rectify inconsistencies, missing values, and outliers in the dataset.
#    - Create new variables or columns to enhance the richness of the dataset.
# 
# 2. **Exploratory Data Analysis (EDA):**
#    - Explore the relationship between room types and pricing.
#    - Identify key factors influencing pricing decisions.
#    - Analyze variations in user traffic among different geographical areas.
# 
# 4. **Visualization:**
#    - Utilize data visualization techniques to present key findings in an accessible and insightful manner.
# 
# 
# This project aims to provide a cleaner, more coherent dataset and deliver valuable insights that can inform strategic decisions for both hosts and the Airbnb platform.

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Read the data set

airbnb_df = pd.read_csv("./Data/Airbnb_Open_Data.csv",low_memory=False)
airbnb_df.head(5)


# In[3]:


# Description

airbnb_df.describe()


# In[4]:


airbnb_df.info()


# In[5]:


# Column Headers

airbnb_df.columns


# In[6]:


# Cheaning Column Headers

airbnb_df.columns = [x.lower().replace(' ','_') for x in airbnb_df.columns]
airbnb_df.columns


# In[7]:


# Shape
airbnb_df.shape


# In[8]:


# droping duplicates

airbnb_df.drop_duplicates(inplace=True)
airbnb_df.shape


# In[9]:


# Dropped about more than 500 duplicate rows


# In[10]:


airbnb_df.info() 


# In[11]:


# To check for count of missing values

airbnb_df.isnull().sum()


# In[12]:


airbnb_df.nunique()


# ## Data Cleaning

# #### Based on observation,
# ##### Columns to be droped:
# 
# 'name', 
# 'host_name', 
# 'lat', 
# 'long', 
# 'country', 
# 'country_code', 
# 'instant_bookable', 
# 'house_rules', 
# 'license'
# 
# ##### Columns to Calculate the values:
# 
# 'host_identity_verified' (289 null values), 
# 'neighbourhood_group' (29 null values), 
# 'neighbourhood' (16 null values), 
# 'cancellation_policy' (76 null values), 
# 'construction_year' (214 null values), 
# 'price' (247 null values), 
# 'service_fee' (273 null values), 
# 'minimum_nights' (400 null values), 
# 'number_of_reviews' (183 null values), 
# 'last_review' (15832 null values), 
# 'reviews_per_month' (15818 null values), 
# 'review_rate_number' (319 null values), 
# 'calculated_host_listings_count' (319 null values), 
# 'availability_365' (448 null values)

# In[13]:


# To Drop the unnecessary columns

airbnb_df.drop([col for col in ['name', 'host_name', 'lat', 'long', 'country', 'country_code', 'instant_bookable', 'house_rules', 'license'] if col in airbnb_df.columns], axis=1, inplace=True)
airbnb_df.shape


# In[14]:


# Calculate the missing values

for i in airbnb_df.columns:
    if i == 'host_identity_verified':
        airbnb_df.loc[airbnb_df[i].isnull(), 'host_identity_verified'] = airbnb_df[i].mode()[0] # Mode
    
    if i == 'neighbourhood':
        airbnb_df.loc[airbnb_df[i].isnull(), 'neighbourhood'] = airbnb_df[i].mode()[0] # Mode
    
    if i == 'cancellation_policy':
        airbnb_df.loc[airbnb_df[i].isnull(), 'cancellation_policy'] = airbnb_df[i].mode()[0] # Mode
        
    if i == 'price':
        airbnb_df.loc[airbnb_df[i].isnull(), 'price'] = airbnb_df[i].mode()[0] # Mode
    
    if i == 'service_fee':
        airbnb_df.loc[airbnb_df[i].isnull(), 'service_fee'] = airbnb_df[i].mode()[0] # Mode
    
    if i == 'construction_year':
        airbnb_df.loc[airbnb_df[i].isnull(), 'construction_year'] = airbnb_df[i].mode()[0] # Mode
        
    if i == 'last_review':
        airbnb_df.loc[airbnb_df[i].isnull(), 'last_review'] = airbnb_df[i].mode()[0] # Mode
    
    if i == 'reviews_per_month':
        airbnb_df.loc[airbnb_df[i].isnull(), 'reviews_per_month'] = airbnb_df[i].mode()[0] # Mode    
    
    if i == 'availability_365':
        airbnb_df.loc[airbnb_df[i].isnull(), 'availability_365'] = airbnb_df[i].mode()[0] # Mode
    
    
    
    if i == 'minimum_nights':
        airbnb_df.loc[airbnb_df[i].isnull(), 'minimum_nights'] = airbnb_df[i].median() # Median

    if i == 'number_of_reviews':
        airbnb_df.loc[airbnb_df[i].isnull(), 'number_of_reviews'] = airbnb_df[i].median() # Median
        
    if i == 'review_rate_number':
        airbnb_df.loc[airbnb_df[i].isnull(), 'review_rate_number'] = airbnb_df[i].median() # Median
    
    if i == 'calculated_host_listings_count':
        airbnb_df.loc[airbnb_df[i].isnull(), 'calculated_host_listings_count'] = airbnb_df[i].median() # Median


# In[15]:


airbnb_df.isnull().sum()


# In[16]:


airbnb_df.nunique()


# In[17]:


neighbourhood_grp = airbnb_df.loc[(airbnb_df['neighbourhood_group'].isnull()),'neighbourhood']
neighbourhood_grp

for i in neighbourhood_grp:
    mode = airbnb_df.loc[(airbnb_df['neighbourhood'] == i) & (airbnb_df['neighbourhood_group'].notnull()),'neighbourhood_group'].mode()[0]
    airbnb_df.loc[(airbnb_df['neighbourhood_group'].isnull()) & (airbnb_df['neighbourhood'] == i), 'neighbourhood_group'] = mode


# In[18]:


airbnb_df['neighbourhood_group'].value_counts()


# In[19]:


airbnb_df.replace({'brookln':'Brooklyn','manhatan':'Manhattan'}, inplace=True)
airbnb_df['neighbourhood_group'].value_counts()


# In[20]:


airbnb_df.isnull().sum()


# In[21]:


for i in airbnb_df.columns:
    print(i, '\n',airbnb_df[i].value_counts(),'*'*40, '\n')


# In[22]:


# https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/code
# https://www.kaggle.com/code/ajii007/airbnb-eda/notebook
# https://www.youtube.com/watch?v=Ky1Jo8th24w
# https://github.com/samujjalp/airbnb-booking-analysis/blob/main/Airbnb_EDA_Project.ipynb


# In[23]:


# Converting float to int

airbnb_df['construction_year'] = airbnb_df['construction_year'].astype(int)
airbnb_df['construction_year']


# In[24]:


# Removing the specical characters and Converting to float. 

airbnb_df['price'] = airbnb_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
airbnb_df['price']


# In[25]:


# Removing the specical characters and Converting to float. 

airbnb_df['service_fee'] = airbnb_df['service_fee'].replace({'\$': '', ',': ''}, regex=True).astype(float)
airbnb_df['service_fee']


# In[26]:


# To remove all the -ve values.

airbnb_df['minimum_nights'] = airbnb_df['minimum_nights'].astype(int)
airbnb_df['minimum_nights'].unique()


# In[27]:


for i in airbnb_df['minimum_nights'].value_counts().index:
    if i < 0:
        airbnb_df['minimum_nights'] = airbnb_df['minimum_nights'].replace(i,-i)
airbnb_df['minimum_nights'].unique()


# In[28]:


airbnb_df['number_of_reviews'] = airbnb_df['number_of_reviews'].astype(int)
airbnb_df['number_of_reviews'].unique()


# Checking for Outliers

# In[29]:


# Function to detect outliers using standard deviation
def detect_outliers_std(data, threshold=3):
    mean = data.mean()
    std = data.std()
    return abs(data - mean) > threshold * std

# Set the threshold for outliers
std_dev_threshold = 3

# Detect outliers for 'price' using standard deviation
outliers_price_std = detect_outliers_std(airbnb_df['price'], threshold=std_dev_threshold)


# Detect outliers for 'service_fee' using standard deviation
outliers_fee_std = detect_outliers_std(airbnb_df['service_fee'], threshold=std_dev_threshold)


# Print the rows with outliers for 'price'
print("Outliers in 'price':")
print(outliers_price_std.value_counts())

# Print the rows with outliers for 'service_fee'
print("\nOutliers in 'service_fee':")
print(outliers_fee_std.value_counts())


# In[30]:


airbnb_df.describe()


# ## Data Visualization

# #### Host Identity Verification:
# 
# What is the distribution of hosts who have verified their identity?

# In[31]:


# Host Identity Verification

verification_distribution = airbnb_df['host_identity_verified'].value_counts()
print("Distribution of Host Identity Verification:")
print(verification_distribution)

plt.figure(figsize=(4,4))
plt.pie(verification_distribution, labels=verification_distribution.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
plt.title('Distribution of Host Identity Verification') 
plt.show()


# #### Neighborhood Characteristics:
# 
# How are listings distributed among different neighborhood groups?
# Which neighborhood group has the highest average price?

# In[32]:


# Neighborhood Characteristics

neighborhood_group_distribution = airbnb_df['neighbourhood_group'].value_counts()
print("Distribution of Listings by Neighborhood Group:")
print(neighborhood_group_distribution)

              
ng = airbnb_df['neighbourhood_group'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=ng.index, y=ng.values)
plt.title('Neighbourhood group')
plt.xlabel('Neighbourhood group Name')
plt.ylabel('Count')
plt.show() 

print("\nAverage Price by Neighborhood Group:")
average_price_by_neighborhood = airbnb_df.groupby('neighbourhood_group')['price'].mean()
print(average_price_by_neighborhood)


# In[33]:


# Get the top 10 neighborhoods

neigh = airbnb_df['neighbourhood'].value_counts().head(25)
neigh

plt.bar(neigh.index, neigh.values)
plt.title('Neighbourhood Top 25 with respect to Counts')
plt.xlabel('Neighbourhood')
plt.ylabel('Counts')
plt.xticks(neigh.index, rotation=90)
plt.show()


# #### Cancellation Policy:
# 
# What are the common cancellation policies adopted by hosts?

# In[34]:


# Cancellation Policy
cancellation_policy_distribution = airbnb_df['cancellation_policy'].value_counts()
print("Distribution of Cancellation Policies:")
print(cancellation_policy_distribution)

canl_pol = airbnb_df['cancellation_policy'].value_counts()

plt.pie(canl_pol.values, labels=canl_pol.index, autopct='%1.2f%%', explode =[0.03,0.03,0.03],startangle=90)
plt.title('Cancellation Policy distribution')
plt.show()


# #### Room Type and Pricing:
# 
# How does the distribution of room types vary in the dataset?
# Is there a significant difference in pricing based on room types?

# In[35]:


# Room Type Distribution and Pricing
room_type_distribution = airbnb_df['room_type'].value_counts()
plt.figure(figsize=(8, 6))

sns.barplot(x=room_type_distribution.index, y=room_type_distribution.values)
plt.title('Distribution of Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show() 


# In[36]:


# Pricing Based on Room Types
room_type_price_comparison = airbnb_df.groupby('room_type')['price'].mean()
print("Average Price by Room Type:")
print(room_type_price_comparison)

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(room_type_price_comparison, labels=room_type_price_comparison.index, autopct='%1.1f%%', startangle=90)
plt.title('Average Price Distribution by Room Type')
plt.show()


# #### Property Construction Year:
# 
# What is the distribution of the construction years of listed properties?

# In[37]:


# Distribution of Property Construction Years
construction_year_distribution = airbnb_df['construction_year'].value_counts()
print("Distribution of Property Construction Years:")
print(construction_year_distribution)

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(construction_year_distribution, labels=construction_year_distribution.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Property Construction Years')
plt.show()

# Plotting a line chart
construction_year_count = airbnb_df.groupby('construction_year')['price'].count()
plt.figure(figsize=(10, 6))
construction_year_count.plot(kind='line', marker='o')
plt.title('Accommodation Count by Construction Year')
plt.xlabel('Construction Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[38]:


# Count of "Type of rooms" Constructed in each Year.
room = airbnb_df.groupby('construction_year')['room_type'].value_counts()
roomDF = pd.DataFrame(room)
roomDF


# #### Pricing and Service Fee:
# 
# How is the pricing structured in relation to the service fee?

# In[39]:


# Pricing Structure in relation to Service Fee.
plt.figure(figsize=(10, 6))

plt.scatter(x='service_fee', y='price', data=airbnb_df.head(200))
plt.title('Pricing Structure in relation to Service Fee')
plt.xlabel('Service Fee')
plt.ylabel('Price')
plt.show()


# In[40]:


# Pricing Structure in relation to Service Fee in each neighbourhood_group.

neighbour = airbnb_df['neighbourhood_group'].unique()

for i in neighbour:
    neighbour_data = airbnb_df.loc[airbnb_df['neighbourhood_group'] == i, ['price', 'service_fee']]
    print(i)
    
    plt.figure()
    
    plt.scatter(neighbour_data['service_fee'],neighbour_data['price'])
    
    plt.title(f'Price vs Service Fee for {i}')
    plt.ylabel('Price')
    plt.xlabel('Service Fee')
    plt.show()


# #### Minimum Nights and Availability:
# 
# What is the typical minimum nights requirement for bookings?
# Is there any pattern between the minimum nights and listing availability?

# In[41]:


# Typical Minimum Nights Requirement for Bookings

typical_min_nights = airbnb_df['minimum_nights'].median()
print(f"Typical Minimum Nights Requirement for Bookings: {typical_min_nights}")


# In[42]:


# Relationship Between Minimum Nights and Availability
plt.figure(figsize=(10, 6))
sns.scatterplot(x='minimum_nights', y='availability_365', data=airbnb_df.head(200))
plt.title('Relationship Between Minimum Nights and Availability')
plt.xlabel('Minimum Nights')
plt.ylabel('Availability in Days')
plt.show()

