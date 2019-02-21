from taskdl.TaskWrapper import TaskWrapper


if __name__ == '__main__':
    TaskWrapper('c9149bf9-01b9-46ea-9218-276ff4ef1192', env='dev').run_task('examples/pytorch_example.py',
                                                                            image='pytorch',
                                                                            task_name='Pytorch example',
                                                                            upload_file_paths=['examples/mnist_train.csv'],
                                                                            model_path='model_saved')