import subprocess


class TaskWrapper(object):

    def __init__(self, dataset_id, workspace='/tmp/', file_delimiter='/', env='us'):
        self.dataset_id = dataset_id
        self.workspace = workspace
        self.file_delimiter = file_delimiter
        self.env = env

    def upload_python_file(self, python_path):
        result = subprocess.check_output(['lo', 'files-upload', python_path, self.dataset_id, '--overwrite'])
        file_id = result.split(' ')[-1]
        return file_id.strip()

    def construct_inputs(self):
        pass

    def construct_body(self, task_name, inputs):
        example = {
            "name": task_name,
            "datasetId": self.dataset_id,
            "gpus": 1,
            "inputs": inputs,
            "outputs": [
                {
                    "path": self.workspace + "my_model",
                    "url": "https://api.%s.lifeomic.com/v1/projects/%s" % (self.env, self.dataset_id),
                    "type": "DIRECTORY"
                }
            ],
            "resources": {
                "cpu_cores": 8,
                "ram_gb": 60
            },
            "executors": [
                {
                    "workdir": "/tmp",
                    "image": "tensorflow/tensorflow:1.12.0-gpu-py3",
                    "command": [
                        "python",
                        "/tmp/TensorflowSparse.py"
                    ],
                    "stderr": "/tmp/test/stderr.txt",
                    "stdout": "/tmp/test/stdout.txt"
                }
            ]
        }


class FileDataset(object):

    def __init__(self, file_id, file_name):
        if not file_id or not file_name:
            raise RuntimeError("File Id and file name must be implemented")
        self.file_id = file_id
        self.file_name = file_name

