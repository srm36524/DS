#!/usr/bin/env python
# coding: utf-8

# # Assignment: SQL Notebook
# 
# 
# 
# ## Introduction
# 
# Using this Python notebook you will:
# 
# 1.  Understand the Spacex DataSet
# 2.  Load the dataset  into the corresponding table in a Db2 database
# 3.  Execute SQL queries to answer assignment questions
# 

# ## Overview of the DataSet
# 
# SpaceX has gained worldwide attention for a series of historic milestones.
# 
# It is the only private company ever to return a spacecraft from low-earth orbit, which it first accomplished in December 2010.
# SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars wheras other providers cost upward of 165 million dollars each, much of the savings is because Space X can reuse the first stage.
# 
# Therefore if we can determine if the first stage will land, we can determine the cost of a launch.
# 
# This information can be used if an alternate company wants to bid against SpaceX for a rocket launch.
# 
# This dataset includes a record for each payload carried during a SpaceX mission into outer space.
# 

# ### Download the datasets
# 
# This assignment requires you to load the spacex dataset.
# 
# In many cases the dataset to be analyzed is available as a .CSV (comma separated values) file, perhaps on the internet. Click on the link below to download and save the dataset (.CSV file):
# 
# <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2021-01-01" target="_blank">Spacex DataSet</a>
# 

# ### Store the dataset in database table
# 
# **it is highly recommended to manually load the table using the database console LOAD tool in DB2**.
# 
# <img src = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/images/spacexload.png">
# 
# Now open the Db2 console, open the LOAD tool, Select / Drag the .CSV file for the  dataset, Next create a New Table, and then follow the steps on-screen instructions to load the data. Name the new table as follows:
# 
# **SPACEXDATASET**
# 
# **Follow these steps while using old DB2 UI which is having Open Console Screen**
# 
# **Note:While loading Spacex dataset, ensure that detect datatypes is disabled. Later click on the pencil icon(edit option).**
# 
# 1.  Change the Date Format by manually typing DD-MM-YYYY and timestamp format as DD-MM-YYYY HH\:MM:SS.
# 
#     Here you should place the cursor at Date field and manually type as DD-MM-YYYY.
# 
# 2.  Change the PAYLOAD_MASS\_\_KG\_  datatype  to INTEGER.
# 
# <img src = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/images/spacexload2.png">
# 

# **Changes to be considered when having DB2 instance with the new UI having Go to UI screen**
# 
# *   Refer to this insruction in this <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Sign%20up%20for%20IBM%20Cloud%20-%20Create%20Db2%20service%20instance%20-%20Get%20started%20with%20the%20Db2%20console/instructional-labs.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2021-01-01">link</a> for viewing  the new  Go to UI screen.
# 
# *   Later click on **Data link(below SQL)**  in the Go to UI screen  and click on **Load Data** tab.
# 
# *   Later browse for the downloaded spacex file.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/images/browsefile.png" width="800"/>
# 
# *   Once done select the schema andload the file.
# 
#  <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/images/spacexload3.png" width="800"/>
# 

# For this exercise, it was difficult to access IBM's Db2 due to high demand. To complete this exercise, we would make use of the postgre SQL.

# In[1]:


# install for connection to postgres database (local)
get_ipython().system('pip install psycopg2')

import csv
import psycopg2
import pandas as pd

# install for connection to the database using DB2 (cloud)
#!pip install sqlalchemy==1.3.9
#!pip install ibm_db_sa
#!pip install ipython-sql
print('Project libraries has been successfully installed!')


# ### Connect to the database using DB2
# 
# Let us first load the SQL extension and establish a connection with the database
# 

# In[2]:


get_ipython().run_line_magic('load_ext', 'sql')


# **DB2 magic in case of old UI service credentials.**
# 
# In the next cell enter your db2 connection string. Recall you created Service Credentials for your Db2 instance before. From the **uri** field of your Db2 service credentials copy everything after db2:// (except the double quote at the end) and paste it in the cell below after ibm_db_sa://
# 
# <img src ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_edX/images/URI.jpg">
# 
# in the following format
# 
# **%sql ibm_db_sa://my-username:my-password\@my-hostname:my-port/my-db-name**
# 
# **DB2 magic in case of new UI service credentials.**
# 
# <img src ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/images/servicecredentials.png" width=600>  
# 
# *   Use the following format.
# 
# *   Add security=SSL at the end
# 
# **%sql ibm_db_sa://my-username:my-password\@my-hostname:my-port/my-db-name?security=SSL**
# 

# In[3]:


get_ipython().run_line_magic('sql', 'ibm_db_sa://')


# In[ ]:





# ### Connect to the database using PostgreSQL database
# 
# We tried to establish connection to the cloud database but experienced some difficulty so we decided to use the postgreSQL database. Let us load the SQL extension and establish a connection with the database using the jupyter notebook.

# In[4]:


# create connection to postgreSQL database
conn = psycopg2.connect(
    host = 'localhost',
    database = 'postgres', 
    user = 'postgres', 
    password = 'chuksoo',  
    port = '5432')
print('Connection to database is successfully')


# In[5]:


# function to read from database
def read(conn, read_):
    print('Read')
    cursor = conn.cursor()
    cursor.execute(read_)
    for row in cursor:
        print(f'row = {row}')
    print()
    
# function to create in postgre database     
def create(conn, create_):
    cursor = conn.cursor() # create cursor object
    cursor.execute(create_) # execute query
    conn.commit() # commit query to database
    print('Table have been created successfull!!!')
    #read(conn)
    
# function to insert in postgre database     
def insert(conn, insert_):
    cursor = conn.cursor()
    cursor.execute(insert_)
    conn.commit()
    print('Records have been successfully inserted!!!')
    #read(conn)
    
# function to update table
def update(conn, update_):
    print('Update')
    cursor = conn.cursor()
    cursor.execute(update_)
    conn.commit()
    #read(conn)
    
# function to delete in postgre database
def delete(conn, delete_):
    print('Delete')
    cursor = conn.cursor()
    cursor.execute(delete_)
    conn.commit()
    #read(conn)

# close the cursor and connection to the server 
def close():
    cursor.close()
    conn.close()   
    
# function to create pandas dataframe
def create_pandas_df(sql_query, database=conn):
    table = pd.read_sql_query(sql_query, database)
    return table


# To begin, we have to create the table in the database using the `CREATE` function. Next we insert the **csv** file by loading the `csv` module. Then we would run the `INSERT` query for each row and then commit the data.

# ### Create table in postgreSQL database

# In[6]:


# create table SpaceX
create_ = '''
            DROP TABLE IF EXISTS SpaceX;
            CREATE TABLE SpaceX
                (
                    Date DATE NULL,
                    Time TIME NULL,
                    BoosterVersion VARCHAR(50) NULL,
                    LaunchSite VARCHAR(50) NULL,
                    Payload VARCHAR(100) NULL,
                    PayloadMassKG INT NULL,
                    Orbit VARCHAR(50) NULL,
                    Customer VARCHAR(100) NULL,
                    MissionOutcome VARCHAR(50) NULL,
                    LandingOutcome VARCHAR(100) NULL
                );
            '''
create(conn, create_)


# If we already have the postgreSQL database open, we would have seen that the table have been created by now. Next we load the csv file using the csv module.

# In[7]:


# loading the csv file
cursor = conn.cursor()
with open('Spacex.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row.
    for row in reader:
        cursor.execute(
        "INSERT INTO SpaceX VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        row
    )
conn.commit()
print('CSV file inserted into database successfully!!!')


# In[8]:


# read to see that data has been uploaded in database
read_ = '''
        SELECT *
        FROM   SpaceX
        LIMIT 10
        '''
read(conn, read_)


# In[9]:


# output postgre query in pandas dataframe
spacex = create_pandas_df(read_, database=conn)
spacex.head(10)


# ## Tasks
# 
# Now write and execute SQL queries to solve the assignment tasks.
# 
# ### Task 1
# 
# ##### Display the names of the unique launch sites  in the space mission
# 

# In[10]:


task_1 = '''
        SELECT DISTINCT LaunchSite 
        FROM SpaceX
'''
create_pandas_df(task_1, database=conn)


# ### Task 2
# 
# ##### Display 5 records where launch sites begin with the string 'CCA'
# 

# In[11]:


task_2 = '''
        SELECT *
        FROM SpaceX
        WHERE LaunchSite LIKE 'CCA%'
        LIMIT 5
        '''
create_pandas_df(task_2, database=conn)


# ### Task 3
# 
# ##### Display the total payload mass carried by boosters launched by NASA (CRS)
# 

# In[12]:


task_3 = '''
        SELECT SUM(PayloadMassKG) AS Total_PayloadMass
        FROM SpaceX
        WHERE Customer LIKE 'NASA (CRS)'
        '''
create_pandas_df(task_3, database=conn)


# ### Task 4
# 
# ##### Display average payload mass carried by booster version F9 v1.1
# 

# In[13]:


task_4 = '''
        SELECT AVG(PayloadMassKG) AS Avg_PayloadMass
        FROM SpaceX
        WHERE BoosterVersion = 'F9 v1.1'
        '''
create_pandas_df(task_4, database=conn)


# ### Task 5
# 
# ##### List the date when the first successful landing outcome in ground pad was acheived.
# 
# *Hint:Use min function*
# 

# In[14]:


task_5 = '''
        SELECT MIN(Date) AS FirstSuccessfull_landing_date
        FROM SpaceX
        WHERE LandingOutcome LIKE 'Success (ground pad)'
        '''
create_pandas_df(task_5, database=conn)


# ### Task 6
# 
# ##### List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
# 

# In[15]:


task_6 = '''
        SELECT BoosterVersion
        FROM SpaceX
        WHERE LandingOutcome = 'Success (drone ship)'
            AND PayloadMassKG > 4000 
            AND PayloadMassKG < 6000
        '''
create_pandas_df(task_6, database=conn)


# ### Task 7
# 
# ##### List the total number of successful and failure mission outcomes
# 

# In[16]:


task_7a = '''
        SELECT COUNT(MissionOutcome) AS SuccessOutcome
        FROM SpaceX
        WHERE MissionOutcome LIKE 'Success%'
        '''

task_7b = '''
        SELECT COUNT(MissionOutcome) AS FailureOutcome
        FROM SpaceX
        WHERE MissionOutcome LIKE 'Failure%'
        '''
print('The total number of successful mission outcome is:')
display(create_pandas_df(task_7a, database=conn))
print()
print('The total number of failed mission outcome is:')
create_pandas_df(task_7b, database=conn)


# ### Task 8
# 
# ##### List the   names of the booster_versions which have carried the maximum payload mass. Use a subquery
# 

# In[17]:


task_8 = '''
        SELECT BoosterVersion, PayloadMassKG
        FROM SpaceX
        WHERE PayloadMassKG = (
                                SELECT MAX(PayloadMassKG)
                                FROM SpaceX
                                )
        ORDER BY BoosterVersion
        '''
create_pandas_df(task_8, database=conn)


# ### Task 9
# 
# ##### List the failed landing_outcomes in drone ship, their booster versions, and launch site names for in year 2015
# 

# In[18]:


task_9 = '''
        SELECT BoosterVersion, LaunchSite, LandingOutcome
        FROM SpaceX
        WHERE LandingOutcome LIKE 'Failure (drone ship)'
            AND Date BETWEEN '2015-01-01' AND '2015-12-31'
        '''
create_pandas_df(task_9, database=conn)


# ### Task 10
# 
# ##### Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order
# 

# In[19]:


task_10 = '''
        SELECT LandingOutcome, COUNT(LandingOutcome)
        FROM SpaceX
        WHERE DATE BETWEEN '2010-06-04' AND '2017-03-20'
        GROUP BY LandingOutcome
        ORDER BY COUNT(LandingOutcome) DESC
        '''
create_pandas_df(task_10, database=conn)

