#Importing libraries
from azureml.core import Workspace
from azureml.core import Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import DatabricksCompute
from azureml.core.compute import ComputeTarget
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import DatabricksStep
from azureml.core.databricks import PyPiLibrary
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

import matplotlib.pyplot as plt

# ### configuring the workspace

# Access the Workspace
ws = Workspace.from_config('/config.json')
ws

# ### Environment creation and register within workspace

# +
# Create the environment
myenv = Environment(name="MyEnvironment")
# Create the dependencies object
myenv_dep = CondaDependencies.create(conda_packages=['scikit-learn', 'joblib', 'matplotlib'])
myenv.python.conda_dependencies = myenv_dep

# Register the environment
myenv.register(ws)
# -

# ### Provisioning configuration using AmlCompute

# +
# Create a compute cluster for pipeline
cluster_name = "pipeline-cluster"
print("Accessing the compute cluster...")

if cluster_name not in ws.compute_targets:
    print("Creating the compute cluster with name: ", cluster_name)
    compute_config = AmlCompute.provisioning_configuration(
                                      vm_size='STANDARD_D2_V2', max_nodes=4)

    compute_cluster = AmlCompute.create(ws, cluster_name, compute_config)
    compute_cluster.wait_for_completion()
else:
    compute_cluster = ws.compute_targets[cluster_name]
    print(cluster_name, ", compute cluster found. Using it...")
    
    
#Create Run Configurations for the steps
run_config = RunConfiguration()  #RunConfiguration object encapsulates the information necessary to submit a training run in an experiment.
run_config.target = compute_cluster
run_config.environment = myenv
# -


# ### Attach the Databricks Cluster as an attached compute target

# +
# Create the configuration information of the cluster
db_resource_group     = "HealthCareX_Azure"
db_workspace_name     = "AL_DBK_0618"
db_access_token       = "dapi029180154f0aefa76ecefec6a80575b8-3"
db_compute_name       = "clusterdbk"


if db_compute_name not in ws.compute_targets:
    print("Creating Configuration for the DB Cluster....")
    attach_config = DatabricksCompute.attach_configuration(
                                resource_group = db_resource_group,
                                workspace_name = db_workspace_name,
                                access_token = db_access_token)
    
    print("Attaching the compute target....")
    db_cluster = ComputeTarget.attach(ws, 
                                      db_compute_name, 
                                      attach_config)
    
    db_cluster.wait_for_completion(True)

else:
    print("Compute target exists...")
    db_cluster = ws.compute_targets[db_compute_name]
# -

# ### Accessing blobstore from azureml

# Create input data reference
data_store = ws.get_default_datastore()

data_store

input_data = DataReference(datastore = data_store,
                           data_reference_name = 'input')

output_data1 = PipelineData('testdata', datastore=data_store)

# pass data reference of Input and Output to databricks
input_data , output_data1

# ### Creating the Databricks configuration

scikit_learn = PyPiLibrary(package = 'scikit-learn')
joblib       = PyPiLibrary(package = 'joblib')


notebook_path = r"/Users/vsekar@cloudseclab2.com/diabetes_pipeline/data_preparation"

db_step01 = DatabricksStep(name = "AzureDataBricks_ML",
                           inputs = [input_data],
                           outputs = [output_data1],
                           num_workers = 1, 
                           notebook_path = notebook_path,
                           run_name = "databricks_notebook",
                           compute_target = db_cluster, #clusterdbk
                           pypi_libraries = [scikit_learn, joblib],
                           allow_reuse = False)


db_cluster

db_step01

# ### Creating the PythonScriptStep configuration

eval_step = PythonScriptStep(name='Model_Evaluation_Metrics',
                                 source_directory='./model_evaulation',
                                 script_name='Evaluate.py',
                                 inputs=[output_data1],
                                 runconfig=run_config,
                                 arguments=['--testdata', output_data1])

eval_step 


# ### Creating the pipeline to run

#Combining azure dbk & azure ml call functions into steps
steps = [db_step01, eval_step]


#Calling the Pipeline function
new_pipeline = Pipeline(workspace=ws, steps=steps)

#Submitting pipeline into experiment
new_pipeline_run  = Experiment(ws, 'DataBricks_AzureMachineLearning_Demo').submit(new_pipeline)

# pipeline trigger and for completi
new_pipeline_run.wait_for_completion(show_output=True)



