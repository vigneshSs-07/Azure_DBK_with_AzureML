# Azure_DBK_with_AzureML

Please find the dataset used in this demo in below link.

https://www.kaggle.com/saurabh00007/diabetescsv

**data_preparation.ipynb** is the databricks jupyter notebook. Cluster creation and related details are explained in the article.

**DatabricksStep_Job.py** and **Evaluate.py** are script files which run in azure ml as SDK.


Summary Steps :
1.	Create a Databricks workspace.
2.	Create Azure machine learning workspace with same subscription, resource group and location as in databricks workspace.
3.	In Databricks UI, link the same with Azure Machine Learning Studio. Refer step no.2
4.	Create Storage account, Blob Storage container (is used for this demo). But other data sources like ADLS, cosmos DB, SQL database, MySQL database can also be used.
5.	The main script for running this pipeline will be present in azure ml.
6.	This pipeline is built in two steps.
a.	Data preparation and model building in azure databricks with the help of mlflow. Same can be done using automl as well.
b.	Metrics for evaluating the model performance as a python script is present in azure ml.
7.	Adding both the steps into azure ml pipeline step.
8.	Execute the pipeline.
9.	Track model performance with different metrics logged, model registry and can be monitored in azure ml.
