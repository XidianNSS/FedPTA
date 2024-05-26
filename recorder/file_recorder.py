import os
import loguru
from fedsimu.core.algorithms.base.recordable import Recordable
from fedsimu.core.algorithms.base.base_server import BaseServer
import shutil
import pickle
from .base_recorder import BaseRecorder
from typing import List
import platform


class FileRecorder(BaseRecorder):
    def __init__(self, logger: loguru._Logger = loguru.logger):
        super().__init__(logger=logger)

    def save_experiment(self,
                        container_records: List[dict],
                        save_path: str = None,
                        name: str = None,
                        additional_info: dict = {}):
        """

        A saved object(dict) should at least have:
            'server_info': server_record,
            'client_info': [client_record_1, client_record_2, ...],
            'name': experiment_name

        :param additional_info: Additional info to be saved with experiment
        :param save_path:
        :param container_records: List[dict]  All records from containers.
        :param name: str  Experiment name.
        :return:

        """
        if save_path is None:
            self.logger.error('Save path cannot be None.')
            raise TypeError

        if name is None:
            abs_work_directory = os.path.abspath(save_path)
            system = platform.platform().lower()
            if 'windows' in system:
                name = abs_work_directory.split('\\')[-1]
            elif "macos" in system:
                name = abs_work_directory.split('/')[-1]
            elif "linux" in system:
                name = abs_work_directory.split('/')[-1]
            else:
                self.logger.warning(f"Unknown type system ({system}), apply experiment name = 'untitled'")
                name = 'untitled'

        self.logger.info("Saving experiment info.")
        experiment_info = {
            'server_info': None,
            'client_info': [],
            'name': name
        }
        experiment_info.update(additional_info)

        byzantine_attacker_nums = 0
        byzantine_attackers = []
        byzantine_attacker_idx = []
        for container_record in container_records:
            if container_record['server_container']:
                assert len(container_record['member_records']) == 1
                experiment_info['server_info'] = container_record['member_records'][0]

            else:
                for record in container_record['member_records']:
                    experiment_info['client_info'].append(record)
                    if 'attacker' in record.keys() and record['attacker']:  # gather attackers info
                        byzantine_attacker_nums += 1
                        byzantine_attackers.append(f"Client{record['client_id']}")
                        byzantine_attacker_idx.append(record['client_id'])
        experiment_info['client_nums'] = len(experiment_info['client_info'])
        experiment_info['rounds'] = experiment_info['server_info']['rounds']
        if byzantine_attacker_nums != 0:
            experiment_info['byzantine_experiment'] = True  # record byzantine related info
            experiment_info['attacker_nums'] = byzantine_attacker_nums
            experiment_info['attackers'] = byzantine_attackers
            experiment_info['attacker_idx'] = byzantine_attacker_idx

        experiment_info_file = save_path
        save_object(experiment_info_file, experiment_info)
        self.logger.info("Experiment info saved.")

    def load_experiment(self, load_path: str = None) -> dict:
        if load_path is None:
            self.logger.error('Save path cannot be None.')
            raise TypeError

        self.logger.info("Loading experiment info.")
        experiment_info_file = load_path
        experiment_info = load_object(experiment_info_file)

        assert isinstance(experiment_info, dict)
        self.logger.info("Experiment info loaded.")
        return experiment_info


def save_object(file_path: str, obj):
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)  # delete with -r if it is a dir
        else:
            os.remove(file_path)
    with open(file_path, 'xb') as f:
        pickle.dump(obj, f)


def load_object(file_path: str):
    obj = None

    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def create_directory(directory_path: str):
    """
    Create directory if not exist. If path exist and the path is directed to a file,
    delete it and create directory
    """
    if not os.path.exists(directory_path):  # path not exist
        os.makedirs(directory_path)

    elif not os.path.isdir(directory_path):  # path exist but not directory
        os.remove(directory_path)
        os.makedirs(directory_path)


def directory_is_empty(directory_path: str):
    """
    Judge if a directory is empty. return True if empty, else False.
    If path does not exist or the path is directed to a file, return False.

    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        content = os.listdir(directory_path)
        if len(content) == 0:
            return True
        else:
            return False
    else:
        return False


def delete_all_files(directory_path: str):
    """
    Delete all files under a directory.
    """

    if (not directory_is_empty(directory_path)) and \
            (os.path.exists(directory_path) and os.path.isdir(directory_path)):
        shutil.rmtree(directory_path)
        create_directory(directory_path)
