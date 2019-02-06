import subprocess
import json
from subprocess import Popen, PIPE

class TaskWrapper(object):

    images = {
        'tensorflow': 'lifeomic/lifeomic_tf_task:latest'
    }

    def __init__(self, dataset_id, workspace='/tmp', file_delimiter='/', env='us'):
        self.dataset_id = dataset_id
        self.workspace = workspace
        self.file_delimiter = file_delimiter
        self.env = env

    def __upload_python_file(self, python_path):
        try:
            result = subprocess.check_output(['lo', 'files-upload', python_path, self.dataset_id, '--overwrite'])
            file_id = str(result, 'utf-8').split(' ')[-1]
            print("File id for python file:")
            print(file_id)
            return file_id.strip()
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError()

    def __upload_data_files(self, file_paths):
        file_ids = []
        try:
            for file_path in file_paths:
                result = subprocess.check_output(['lo', 'files-upload', file_path, self.dataset_id, '--overwrite'])
                file_id = str(result, 'utf-8').split(' ')[-1].replace('\n', '')
                print("File id for dataset file:")
                print(file_id)
                file_ids.append(file_id)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError()

        return file_ids

    def __construct_inputs(self, python_path, upload_file_paths=None, file_datasets=None):
        file_name = python_path.split(self.file_delimiter)[-1]
        file_id = self.__upload_python_file(python_path)
        inputs = [{
            "path": '%s/%s' % (self.workspace, file_name),
            "url": "https://api.%s.lifeomic.com/v1/files/%s" % (self.env, file_id),
            "type": "FILE"
        }]

        if upload_file_paths:
            file_ids = self.__upload_data_files(upload_file_paths)
            file_names = [file_path.split(self.file_delimiter)[-1] for file_path in upload_file_paths]
            for i in range(len(file_names)):
                inputs.append({
                    "path": '%s/%s' % (self.workspace, file_names[i]),
                    "url": "https://api.%s.lifeomic.com/v1/files/%s" % (self.env, file_ids[i]),
                    "type": "FILE"
                })

        if file_datasets:
            for file_d in file_datasets:
                data_type = "DIRECTORY" if file_d.is_directory else "FILE"
                inputs.append({
                    "path": '%s/%s' % (self.workspace, file_d.file_name),
                    "url": "https://api.%s.lifeomic.com/v1/files/%s" % (self.env, file_d.file_id),
                    "type": data_type
                })
        return inputs

    def construct_body(self, task_name, inputs, image, output_path='model_data.zip'):
        return {
            "name": task_name,
            "datasetId": self.dataset_id,
            "inputs": inputs,
            "outputs": [
                {
                    "path": '%s/%s' % (self.workspace, output_path),
                    "url": "https://api.%s.lifeomic.com/v1/projects/%s" % (self.env, self.dataset_id),
                    "type": "DATASET"
                }
            ],
            "resources": {
                "gpu_cores": 1
            },
            "executors": [
                {
                    "workdir": self.workspace,
                    "image": image,
                    "command": [
                        "python",
                        inputs[0]['path']
                    ],
                    "stderr": "%s/test/stderr.txt" % self.workspace,
                    "stdout": "%s/test/stdout.txt" % self.workspace
                }
            ]
        }

    def run_task(self, python_path, image='tensorflow', task_name='Deep Learning Task', file_datasets=None,
                 upload_file_paths=None):
        image = self.images[image]
        inputs = self.__construct_inputs(python_path, upload_file_paths, file_datasets)

        body = self.construct_body(task_name, inputs, image)
        task_json = json.dumps(body)
        print("Starting Task")
        print(task_json)
        path = 'tmp_task.json'
        try:
            with open(path, 'w') as tmp:
                tmp.write(task_json)
            p1 = Popen(['cat', path], stdout=PIPE)
            p2 = Popen(["lo", 'tasks-create'], stdin=p1.stdout, stdout=PIPE)
            p1.stdout.close()
            result = p2.communicate()[0]
            print(str(result, 'utf-8'))
            print("Finished Task")
        except subprocess.CalledProcessError as e:
            print(e.output)


class FileDataset(object):

    def __init__(self, file_id, file_name, is_directory=False):
        if not file_id or not file_name:
            raise RuntimeError("File Id and file name must be implemented")
        self.file_id = file_id
        self.file_name = file_name
        self.is_directory = is_directory


