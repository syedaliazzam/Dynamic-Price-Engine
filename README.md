




#  Dynamic Pricing

Scope of this project is to design a pricing model that would determine a customer’s propensity to pay and charge them accordingly while also catering to the supply/demand gap against each day of the week and hour of the day. This document covers Both the modules of this project dealing with dynamic pricing on the customer level. There are two seperate code files for the two modules.


# module 1

The implementation comprises 2 descriptive and 1 predictive model. Results from both the descriptive models elaborating on a customer's transactional and demographic value are fed into the machine learning model to determine their propensity to pay and a surge multiple respective to each customer. 

**RFM Analysis**(Determines Transactional Behavior by scoring customers on Recency, Frequency and Monetary values of their trips)
**POI Modeling** (Infers Profession of Customers based on Frequency of Morning Trips to a specific Building POI) 
**Machine Learning** (Neural Network based Prediction of Customer’s Propensity to Pay and fare multiplier) 


With the aim of making this a supervised classification problem, customers are pre-labeled to train the model using the following definition.


# Determining Propensity To Pay

A customer’s willingness and ability to pay is gauged acoss 7 conditions and predictions are made on the 30 day active population against their lifetime behavior including Fulfillment, Cross Category Utilization, Promo Utilization and participation in historical marketing quest campaigns. 2.64% of our customers fulfilled the following conditions and this population imbalance is catered through oversampling in the machine learning phase.

**Frequency Score** = 3/4 (Last 30 Days)
**Monitory Score** = 4 (Last 30 Days)
**Profession** != ‘Student’ (Last 15 days)
**No. of Low Bids** < **No. of HIgh Bids** (Last 30 Days)
**Avg KMs Per Trip** > 5 KM (Since Jan 22’)
**Avg TPU** > 8 (Last 3 Months)

# RFM Analysis

RFM Analysis is a quantitative ranking technique that scores the customers based on the recency, frequency and monetary value of their transactions. Data is distributed in 4 quartiles and the position of each customer with respect to the distribution determines their score which ranges from 1 to 4 with 4 being the highest. The scores are then concatenated and all possible permutations of these concatenated scores are grouped into the following categories:

# Best Customer  
Perfect RFM Score
RFM Score = 444

# Big Value Provider 
Good Recency and frequency, perfect monetary score.
RFM Score = 334/344/434

# Loyal Customers 
Perfect Recency and Frequency
RFM Score = 331/332/333/341/342/343/431/432/433/441/442/443

# Need Attention 
Average Recency and Frequency 
RFM Score =231/232/233/234/241/242/243/244/311/312/313/313/314/321/322/323/324/411/412/413/414/421/422/423/424

# About to Sleep 
Poor Recency 
RFM Score = 121/122/123/124/131/132/133/134/141/142/143/144/211/212/213/214/221/222/223/224

# Lost 
Poor RFM Score
RFM Score = 111/112/113/114


# POI Modeling



**School 
College
University
Commercial Building
Office
Industrial 
Stadium
Sports Center
Mall
Hospital
Government Office
**

Customers who took trips between 7:00 am to 11:00 am in the last 15 days (23rd Oct 22 - 07th Nov 22) are considered to distinguish morning users under the assumption that people travel to work or for education during these hours and the nearest POI is tagged with each trip’s dropoff location. The coordinates with the highest frequency against each customer is considered as their primary dropoff location across trips and a profession is tagged to that corresponding POI. In the case of multiple POIs having the same frequency, multiple categories are tagged to the customers. Following is the distribution of the major segments identified:


As this exercise was initially conducted in march, Abbas made calls to a sample of these tagged customers and identified that in most cases ‘Students’ were being tagged accurately but ‘Working Professionals’ had cases of mistagging. Taking that into consideration, our outcome definition for dynamic pricing includes ‘Not being a student’ as a factor instead of considering ‘being a working professional’ to have more propensity to pay.  


# Machine Learning Model

Results from RFM and POI Modeling are used to determine our outcome definition which also includes average trips, avg KMs of customers and the frequency of high and low bids in the last 30 days. To determine the behavior of customers against this definition, following data points reflecting the customer’s historical trends were considered.


# Data Points
**Days since Last TXN
Net Bookings Last 30 Days
Gross Bids
Net Bids
Lifetime Net Bookings
Lifetime Promo Trips
Acquisition Source
Age of Platform
Avg PC1 Commision per Trip
Frequent Morning Pickup Zone 
Frequent Evening Drop Off Zone
Targeted In Marketing TPU Campaigns (Yes/No)
Targeted In Marketing ETA Campaigns (Yes/No)**

# MLP Neural Network

**Imbalanced Class Problem**

Since our definition involves 7 conditions, only 2.64% of our customers were initially labeled to have a propensity to pay which means that our algorithm would have been unable to identify the trends of the minority class. To cater to this issue, an oversampling technique called SMOTE was used. SMOTE creates synthetic (not duplicate) samples of the minority class. Hence making the minority class equal to the majority class. SMOTE does this by selecting similar records and altering that record one column at a time by a random amount within the difference to the neighboring records.

# Abstract

WIth the aim of determining a customer’s propensity to pay and apply dynamic pricing, we have implemented a combination of techniques to overcome our lack of demographic information pertaining to the customer. The RFM model gauges the transactional value of customers while frequency based POI tagging of customers allows us to infer their professions. Pairing this information with the historical trip level data of customers enables us to predict their propensity to pay and the probability of that outcome. 

Based on this probability, a multiple is determined against each customer which would be added in the distance based estimated fare for each trip. This multiple would range from 0.01 to 1 meaning that the maximum surge a customer can incur would be 2X. However, we’ll fix the upper limit to 1.5X and treat all customers with above 50% probability the same way. 


# Module 2

# Methodology


We have used a hierarchical approach for clustering using Machine learning,This Algorithm was used because it gives the user the most control over the clusters and provides a clearer picture of the clusters.One Drawback is that it can not be used on Big Data as the computational Power is very big.This will help us in setting a different multiplier based on the demand and supply in a specify zone, hour and day.Agglomerative hierarchical clustering (AHC) is a clustering method that merges two objects or nearby clusters into one cluster, recursively, so that one big cluster that contains all the objects in the data is formed

Aggolomerative approach for Hierarchical clustering using Ward’s Method and Average linkage is used in the Model.


# Multiplier Matrices


To determine the Multiplier we have grouped the data into different clusters based on 5 different Matrices.These Matrices define the supply and demand.Data is collected over a span of 30 Days.The data is grouped by Day,Hour and Zone.

**Gross Trips** : Average Gross Trips  
**Gross Customer** : Average Gross Unique Customers 
**Fulfillment** : NET/Gross 
**Real Fulfillment**:   Net trips / Unique Gross Customers
**Active Drivers** : Average Drivers who received Ringer
**Expired Trips**: Average Expired Trips
**Canceled Trips**: Average Canceled Trips

# Machine Learning Model

Agglomerative hierarchical clustering (AHC) is a clustering method that merges two objects or nearby clusters into one cluster, recursively, so that one big cluster that contains all the objects in the data is formed.There are two parameters that are used to measure the proximity between two objects or clusters, namely the distance measure and the linkage method. Distance measure is used to measure the proximity between two objects or clusters which contain only two objects if it is merged. The linkage method is used to measure the proximity between two clusters which when merged the number of objects is more than two. Output of AHC is a dendrogram which is a hierarchy diagram that summarizes the process of clustering. T.

The model uses euclidean distance between data points to create clusters.We have  used two methods  to create clusters,Average linkage method takes the average distance between all points and clusters points with minimum distance.This Method is strong against outliers in the data as outlier effect is canceled by averaging out the distance.Wards Method takes all points as clusters and then takes the centroid of the two nearest clusters and uses the centroid as the new point of computation for the next cluster formation.


Input: A set 𝑋 of objects {𝑥1, … , 𝑥𝑛}, distance function 𝑑𝑖𝑠𝑡(𝑐1, 𝑐2)
 Procedure: for 𝑖 = 1 to 𝑛
              𝑐1 = {𝑥𝑖 } 
        end for 
        𝐶 = {𝑐1, … , 𝑐𝑛} 
        𝐼 = 𝑛 + 1
                   while 𝐶. 𝑠𝑖𝑧𝑒 > 1 do
                            (𝑐𝑚𝑖𝑛1, 𝑐𝑚𝑖𝑛2 ) =minimum 𝑑𝑖𝑠𝑡(𝑐𝑖 , 𝑐𝑗)for all 𝑐𝑖 , 𝑐𝑗 in 𝐶
   Remove 𝑐𝑚𝑖𝑛1 and 𝑐𝑚𝑖𝑛2 from 𝐶 Add {𝑐𝑚𝑖𝑛1, 𝑐𝑚𝑖𝑛2} to 
𝐶 𝐼 = 𝐼 + 1 end while 


# Abstract 

Dynamic Pricing is a mode which brings together Demand and Supply based upon individual Customer level and Geo location. The problem is how to compute the equilibrium price with the large number of participants in short a need for a self learning model. For 𝑚 driver 𝑛 rider at the same time period,in a specific location  there is a need for varying prices based upon the characteristics of customer and Location. This Document  proposed Agglomerative Hierarchical Clustering (AHC) to be applied for determining clusters of Zones with the same characteristics. By applying AHC, the zone which is not feasible based on their day and time is proposed to be a Multiplier  in price.

To decide pairs that will have a higher Multiplier  Real FF, Gross Trips and are taken into account. As a result, we obtain a number of optimum clusters.The multiplier will be  applied against each trip.The multiplier will range from 5% to 25%.


