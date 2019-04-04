from taskdl.TaskWrapper import TaskWrapper


if __name__ == '__main__':
    TaskWrapper('19e34782-91c4-4143-aaee-2ba81ed0b206', env='dev').run_task('examples/VariantTaskExample.py',
                                                                            task_name='Variant Task',
                                                                            upload_file_paths=['examples/variant_data.json'],
                                                                            cohort_path='variant_model/cohort.csv',
                                                                            model_path='variant_model.zip')