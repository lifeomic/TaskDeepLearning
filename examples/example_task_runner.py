from taskdl.TaskWrapper import TaskWrapper


if __name__ == '__main__':
    TaskWrapper('c9149bf9-01b9-46ea-9218-276ff4ef1192', env='dev').run_task('examples/language_example.py',
                                                                            upload_file_paths=['examples/imdb.npz'],
                                                                            model_path='language_model.zip')