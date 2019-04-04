# Lifeomic Task Service Deep Learning Tools

At Lifeomic, we make it easy to train deep learning models on our task service. This project serves as a tool to run these 
models using only python code. We have included several examples, and below, we will discuss how to get up and running with 
any GPU job on task service.

### Installing

In order to use this tool, you must first have our lifeomic cli installed [which is located here](https://github.com/lifeomic/cli).
After this has been globally installed, you can install `taskdl` through pip:

```
pip install taskdl
```


### Getting Started with Submitting a task

If you already have your python code to train a deep learning model, you can use our utility class called `TaskWrapper` 
to submit job to task service. First, lets look at an example to explain what is happening.

```python
from taskdl.TaskWrapper import TaskWrapper
TaskWrapper('19e34782-91c4-4143-aaee-2ba81ed0b206').run_task('examples/VariantTaskExample.py',
                                                            task_name='Variant Task',
                                                            upload_file_paths=['examples/variant_data.json'],
                                                            cohort_path='variant_model/cohort.csv',
                                                            model_path='variant_model.zip')
```

In this example, the first argument in the constructor is the projectId. This is the LifeOmic project where you want to run 
the task and upload files. It will also be where the output of the task is saved as well. You will need read and write permissions 
to run a task with this library.

When looking at the method `run_task`, we can see a few different options. The first argument specifies the path of the 
python file. This is where your code will live. The `task_name` parameter specifies what you want to call the task in 
LifeOmic. The `upload_file_paths` parameter allows you to upload data from your local machine and automatically get added 
to the task's base directory.

Finally, the last two parameters are relate to the path of the running task. For example, in `cohort_path`, this specifies 
the CSV path for cohorts on the running task. This path is in reference to where the running directory is located. For example, 
if you are running a task with the base directory of `/tmp/` and you save a cohort to `/tmp/cohort.csv`, then for the cohort_path, 
you only need to input `cohort.csv`. It is worth noting that this will create a LifeOmic cohort when finishing a task.

The `model_path` parameter declares where the saved model will be. Typically, you will use a zip file here. In later documentation, 
we will explain some of the utilities we offer to help make this a seamless process for the user.

