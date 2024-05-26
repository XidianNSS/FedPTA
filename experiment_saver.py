import loguru
from fedsimu.reporter.basic_reporter import BasicReporter
from fedsimu.recorder.file_recorder import FileRecorder, create_directory, delete_all_files, load_object
from typing import List
import os


class ExperimentSaver(object):
    def __init__(self,
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger

    def save_experiment(self,
                        container_records: List[dict],
                        root: str = None,
                        name: str = 'untitled',
                        additional_info: dict = {}):
        if root is None:
            root = './' + name.split(' ')[0]

        create_directory(root)
        root_abs = os.path.abspath(root)
        delete_all_files(root_abs)

        create_directory(root_abs + '/report')
        report_abs = os.path.abspath(root_abs + '/report')

        recorder = FileRecorder(logger=self.logger)
        recorder.save_experiment(container_records, save_path=root_abs + '/record.pkl',
                                 name=name, additional_info=additional_info)
        experiment_info = recorder.load_experiment(root_abs + '/record.pkl')

        reporter = BasicReporter(logger=self.logger)
        reporter.generate_abstract(experiment_info, save_path=report_abs + '/abstract.txt')
        reporter.plot_test_acc_loss(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_local_distribution(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_train_acc_loss(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_byzantine_defense(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_cluster(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_asr(experiment_info, show=False, save=True, save_path_root=report_abs)

    def reload_report(self, root: str):
        recorder = FileRecorder(logger=self.logger)

        root_abs = os.path.abspath(root)
        create_directory(root_abs + '/report')
        report_abs = os.path.abspath(root_abs + '/report')
        delete_all_files(report_abs)

        experiment_info = recorder.load_experiment(root_abs + '/record.pkl')
        reporter = BasicReporter(logger=self.logger)
        reporter.generate_abstract(experiment_info, save_path=report_abs + '/abstract.txt')
        reporter.plot_test_acc_loss(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_local_distribution(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_train_acc_loss(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_byzantine_defense(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_cluster(experiment_info, show=False, save=True, save_path_root=report_abs)
        reporter.plot_asr(experiment_info, show=False, save=True, save_path_root=report_abs)



