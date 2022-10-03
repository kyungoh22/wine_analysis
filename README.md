# Blueberry Winery 

- This was the first project done as part of the Code Academy Berlin bootcamp in data science

### Premise of the project

- BlueBerry Winery's team, a start-up wine maker in Portugal, has approached us to help them build a Wine Quality Analytics System. The goal is to determine the quality of the wines produced based on their composition.
- We have separate datasets for white and red wine samples. 
- Each row provides us with the chemical compositions of the wine, as well as the quality rating (out of 10) given by a wine expert. 

### Project structure

#### Part 1: EDA
- Please see the file **EDA.ipynb**
- Creating various plots to visualise the correlation between individual features (chemical properties) and wine quality


#### Part 2: Transformation of features
- Please see the file **gaussian_transformation.ipynb**
- I tried to make the distributions of the features more Gaussian using different transformations in preparation for the building of machine learning models


#### Part 3: Machine learning models
- Please see the file **ml_models.ipynb**
- I build ML models to predict wine quality based on chemical composition
- Models include: logistic regression, SVM, decision trees, random forest, KNN
- The data is very imbalanced. The majority of the samples are in the "medium" quality category. Very few samples are of "low" and "high" quality. As such, I also explore under- and over-sampling to address this imbalance. 
- **NOTE** Much of the parameter tuning was done in a separate notebook / environment. This process was not included in the final notebook in order to save space.


