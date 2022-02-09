# H1N1 and Seasonal Flu Vaccine Predictions
![fluvirus](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/flu_banner.jpeg)

## Project Overview

We have been hired by the CDC as a part of a campaign to increase awareness and encourage more individuals to get vaccinated. We will assist them in making recommendations of how to increase awareness and identify which factors most influence people to get vaccinated.

### Business Problem

Which factors most influence whether a person receives a seasonal or H1N1 flu vaccine?

### Recommendations

Based on our analysis of the data, we would recommend honing in on these 4 factors:
<ul>
  <li>Doctor recommendation of vaccines</li>
  <li>Opinion of vaccine effectiveness</li>
  <li>Opinion of flu risk</li>
  <li>Health insurance</li>
</ul>

We will dive into this further below!

### Guiding Questions for Predictive Modeling

We used these 4 quesitons below as a guide to completing this analysis:
<ul>
  <li>Which model is the most effective in predicting whether a person received a seasonal flu shot?</li>
  <li>Which model is the most effective in predicting whether a person received a H1N1 flu shot?</li>
  <li>Which features are the most important in predicting whether a person received a seasonal flu shot?</li>
  <li>Which features are the most important in predicting whether a person received a H1N1 flu shot?</li>
</ul>



# The Data
![databanner](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/data-science-banner.jpeg)

The data for our study is obtained from the Drivendata https://www.drivendata.org/competitions/66/flu-shot-learning/data/National. This was collected in 2009 H1N1 Flu Survey (NHFS) which was held by the Centres for Disease Control and Prevention (CDC). The survey sample consiste of over 26,000 people. This survey asked people whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. This data was obtained to get a fair idea about the knowledge of people on the effectiveness and safety of flu vaccines and to learn why some people refrained from getting vaccinated against the H1N1 flu and seasonal flu.

The dataset contains the following columns/ information:
<ul>
<li><strong>h1n1_vaccine</strong> - Have they received H1N1 flu vaccine</li>
<li><strong>seasonal_vaccine</strong> - Have they received seasonal flu vaccine</li>
<li><strong>h1n1_concern</strong> - Level of concern about the H1N1 flu on a scale from 0-3</li>
<li><strong>h1n1_knowledge</strong> - Level of knowledge about H1N1 flu on a scale from 0-3</li>
<li><strong>behavioral_antiviral_meds</strong> - Have they taken antiviral medications (binary)</li>
<li><strong>behavioral_avoidance</strong> - Do they avoid close contact with others with flu-like symptoms (binary)</li>
<li><strong>behavioral_face_mask</strong> - Do they have a face mask (binary)</li>
<li><strong>behavioral_wash_hands</strong> - Do they frequently wash their hands (binary)</li>
<li><strong>behavioral_large_gathering</strong>s - Have they avoided large gatherings (binary)</li>
<li><strong>behavioral_outside_home</strong> - Have they reduced contact with people outside of their own home (binary)</li>
<li><strong>behavioral_touch_face</strong> - Do they avoid touching their face (binary)</li>
<li><strong>doctor_recc_h1n1</strong> - Was the H1N1 flu vaccine was recommended by doctor (binary)</li>
<li><strong>doctor_recc_seasonal</strong> - Was the Seasonal flu vaccine was recommended by doctor (binary)</li>
<li><strong>chronic_med_condition</strong> - Do they have chronic medical conditions (binary)</li>
<li><strong>child_under_6_months</strong> - Are they in close contact with a child under the age of six months (binary)</li>
<li><strong>health_worker</strong> - Are they a healthcare worker (binary)</li>
<li><strong>health_insurance</strong> - Do they have health insurance (binary)</li>
<li><strong>opinion_h1n1_vacc_effective</strong> -  What is their opinion about H1N1 vaccine effectiveness on a scale from 1-5</li>
<li><strong>opinion_h1n1_risk</strong> - What is their opinion about risk of getting sick with H1N1 flu without vaccine on a scale from 1-5</li>
<li><strong>opinion_h1n1_sick_from_vacc</strong> - Are they worried about getting sick from taking H1N1 vaccine on a scale from 1-5</li>
<li><strong>opinion_seas_vacc_effective</strong> - What is their opinion about seasonal flu vaccine effectiveness on a scale from 1-5</li>
<li><strong>opinion_seas_risk</strong> - What is their opinion about risk of getting sick with seasonal flu without vaccine on a scale from 1-5</li>
<li><strong>opinion_seas_sick_from_vacc</strong> - Are they worried about getting sick from taking seasonal flu vaccine on a scale from 1-5</li>
<li><strong>age_group</strong> - What age group do they fall into</li>
<li><strong>education</strong> - What is their education level</li>
<li><strong>race</strong> - Race </li>
<li><strong>sex</strong> - Sex </li>
<li><strong>income_poverty</strong> - Household annual income with respect to 2008 Census poverty thresholds</li>
<li><strong>marital_status</strong> - Married or Not Married</li>
<li><strong>rent_or_own</strong> - Housing situation (Rent or Own)</li>
<li><strong>employment_status</strong> - Employment status (Employed or Unemployed)</li>
<li><strong>hhs_geo_region</strong> - Residence using a 10-region geographic classification - values are random character strings</li>
<li><strong>census_msa</strong> - Where is the residence within metropolitan statistical areas (MSA) as defined by the U.S. Census</li>
<li><strong>household_adults</strong> - Number of other adults in household, top-coded to 3</li>
<li><strong>household_children</strong> - Number of children in household, top-coded to 3</li>
<li><strong>employment_industry</strong> - Type of industry they are employed in - values are random character strings</li>
<li><strong>employment_occupation</strong> - Type of occupation - values are random character strings.</li>
</ul>

## Exploratory Data Analysis

We did a very thorough exploratory data analysis on our dataset before making our predictions on our targets. We changed the non-numerical results to numerical, cleaned up the zeros, and made sure our data was ready to begin the analysis! 

There are two target variables in the labels dataset:
<ul>
<li><strong>h1n1_vaccine</strong></li> - Whether respondent received H1N1 flu vaccine.
<li>strong>seasonal_vaccine</strong></li> - Whether respondent received seasonal flu vaccine.
</ul>
Both are binary variables: 0 = No; 1 = Yes. Some respondents didn't get either vaccine, others got only one, and some got both.

### Bar Graphs of Features

We then plotted our features against our targets ('H1N1 Vaccine and Seasonal Vaccine') to see if there are any correlations that need to be considered. Here are the bar graphs of features and their rate based on H1N1 and seasonal survey respondents.
![features](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/featuresgraph.png)

### Heatmap

We created a heatmap to get a better understanding of the correlations between each feature, and their correlations to the H1N1 and seasonal vaccine. 
![heatmap](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/heatmap.png)

## Methodology

After thoroughly cleaning and reviewing the data, we are now ready to build some models! Listed below are the classifiers that we used to create our models:
<ul>
  <li>Logistic Regression</li>
  <li>Decision Tree</li>
  <li>Support Vector Machines</li>
  <li>Naive Bayes</li>
  <li>Voting Classifier</li>
  <li>Random Forest</li>
  <li>Gradient Boost</li>
  <li>Histogram Gradient Boost</li>
  <li>Cat Boost</li>
</ul>

# Business Results

After making a myriad of different models to predict our targets (H1N1/seasonal vaccinations) we found that our best model was the CatBoost classifier using cross validation and a Bayesian optimization tool called Optuna. It outperformed all the models that we ran.

## H1N1 Analysis

Here is a graph showing how our models performed based on the ROCAUC score for H1N1 vaccinations.

![h1n1modelsummary](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/h1n1models.png)

#### ROCAUC Score

The ROCAUC score for our model was .87.
<br>
![h1n1roc](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/h1n1%20roc.png)

#### Confusion Matrix

The confusion matrix reflects how well our model predicted true(1) when true(1) and false(0) when false(0) for H1N1 vaccinations.
![h1n1confusionmatrix](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/h1n1%20cm.png)

#### Permutation Feature Importance

This plot below shows permutation feature importances, which tells us which features were the most important in our model's accuracy. The top features were personal opinions about the virus and vaccinations as well as recommendations by a doctor. 

![h1n1permutations](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/h1n1%20permutation.png)

## Seasonal Analysis

Here is a graph showing how our models performed based on the ROCAUC score for seasonal vaccinations.

![seasonalmodelsummary](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/seasonalmodels.png)

#### ROCAUC Score

The ROCAUC score for our model was .86.
<br>
![seasonalroccurve](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/seasonalroc.png)

#### Confusion Matrix

The confusion matrix reflects how well our model predicted true(1) when true(1) and false(0) when false(0) for seasonal vaccinations. 
![seasonalconfusionmatrix](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/seasonal%20permutation.png)

#### Permutation Feature Importance

Like the H1N1 model, the features that were also important in the model's seasonal vaccination accuracy, are personal opinions and recommendations from a doctor.
![seasonalpermutations](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/seasonalpermutation.png)

#### Impact of differing age groups

Based on the chart below, the opinions of the risks from gettin the flu are about the same across all age groups.

<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%202.29.22%20PM.png" width="500" height="350">


<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%202.29.48%20PM.png" width="500" height="350">
<br>

### Comparing Doctor recommendations to H1N1 & Seasonal vaccinations

As you can see below, there is a large impact on the percentage of people that received vaccinations due to the recommendation from a dr. This is true in both cases for H1N1 and for seasonal flu.

<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%202.36.30%20PM.png" width="500" height="350">
<br>
<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%202.36.21%20PM.png" width="500" height="350">

### A Closer Look at the Health Insurance Feature

As you can see in the chart below, people with health insurance have a higher likelihood of receiving a vaccine. On the other hand, people without insurance are more unlikely to get the vaccine. Individuals without insurance are less likely to see a doctor, so their chances of getting the vaccine recommended to them, falls significantly. As mentioned previous, a doctor's recommendation is one of the top 3 most influential predictors.

<br>
<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%202.34.03%20PM.png" width="500" height="350">
<br>

### Another Feature of Importance - Household Income

In addition to not having health insurance, people that have a household income below the poverty line are also less likely to get vaccinated. 

<br>
<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%202.33.55%20PM.png" width="500" height="350">
<br>

## Classification App

<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%201.46.23%20PM.png" width="1000" height="600">
<br>
<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%201.46.36%20PM.png" width="1000" height="600">
<br>
<img src="https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/Screen%20Shot%202022-01-28%20at%201.46.49%20PM.png" width="1000" height="600">
<br>

# Conclusion

![bottlebanner](https://github.com/AbsIbs/H1N1_flu_vaccine_project/blob/main/images/bottle%20banner.jpeg)

Overall, the 3 most influential factors in determining whether a person got vaccinated against H1N1 or the seasonal flu were:

<ul> 
<li> How effective they believed the vaccine to be at protecting against the flu.</li>
<li> Whether a doctor recommended that they get the vaccine.</li>
<li> Their perceived level of risk of getting sick with the flu without the vaccine.</li>
</ul>

## Recommendations

Based on our analysis, we would make the following 3 recommendations to the CDC to help reach their goal of increasing awareness and increasing the number of vaccinated people:

<ul> 
<li> Doctors should regularly recommend that their patients get vaccinated.</li>
<li> Targeted efforts should be made to inform and provide access to vaccinations for people without health insurance and below the poverty line.</li>
<li> Inform people of the potential risks of H1N1/seasonal flu and provide more education on vaccine effectiveness.</li>
</ul>


