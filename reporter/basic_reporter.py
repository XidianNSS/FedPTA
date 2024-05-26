import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from copy import deepcopy
import loguru
import numpy as np
from typing import List, Union
import pandas as pd
import math
from fedsimu.recorder.file_recorder import delete_all_files, create_directory
from fedsimu.reporter.utils import get_y_distribution_from_dataloaders, \
    get_unique_labels_from_dataloaders


class BasicReporter(object):
    def __init__(self,
                 logger: loguru._Logger = loguru.logger):
        """

        A saved object(dict) should at least have:
            'server_info': server_record,
            'client_info': [client_record_1, client_record_2, ...],
            'name': experiment_name

        """
        self.logger = logger

    def plot_local_distribution(self,
                                experiment_info: dict,
                                show: bool = True,
                                save: bool = False,
                                save_path_root: str = None,
                                client_nums_in_figure: int = 10):
        """
        This function plot local distribution in each client for single experiment.
        :param experiment_info: (dict)  Single experiment info
        :param show: (bool)  run plt.show()
        :param save: (bool)  run plt.save(), need to specify a save_path_root if true.
        :param save_path_root:
        :param client_nums_in_figure: (int) How many client report in single picture.
        :return:
        """
        self.logger.info('Plotting local distribution in each client.')

        if save and (save_path_root is None):
            self.logger.error('Save file is set to True but got no save_path_root.')
            raise TypeError

        # clean directory
        if save:
            create_directory(save_path_root)
            abs_path_root = os.path.abspath(save_path_root)
            save_path_root = abs_path_root

            create_directory(save_path_root + f'/local_distribution')
            delete_all_files(save_path_root + f'/local_distribution')

        client_names = []  # get needed info from records
        client_loaders = []
        for info in experiment_info['client_info']:
            client_names.append(f"client {info['client_id']}")
            client_loaders.append(info['dataloader'])

        unique_labels_list = get_unique_labels_from_dataloaders(client_loaders)
        label_distribution = get_y_distribution_from_dataloaders(client_loaders, client_names, unique_labels_list)

        # Add empty data in dataframe tail
        empty_data_num = (client_nums_in_figure - len(client_names) % client_nums_in_figure) % client_nums_in_figure
        empty_data = {}
        for column in label_distribution.columns:
            empty_data[str(column)] = [np.nan] * empty_data_num
        empty_data = pd.DataFrame(empty_data, index=['NaN'] * empty_data_num)
        label_distribution = pd.concat([label_distribution, empty_data], ignore_index=False)

        # plot figure
        loop_times = math.ceil(len(client_names) / client_nums_in_figure)  # plot
        for i in range(loop_times):
            start_idx = i * client_nums_in_figure
            end_idx = (i + 1) * client_nums_in_figure

            plt.figure()
            label_distribution.iloc[start_idx:end_idx, :].plot.barh(stacked=True)
            plt.xlabel('sample num')
            plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
            if save:
                save_path_fig = save_path_root + f'/local_distribution/client{start_idx}-{end_idx - 1}.png'
                plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

            if show:
                plt.show()
            plt.close()

        save_path = os.path.abspath(save_path_root + f'/local_distribution')
        self.logger.info(f"Local distribution plotted. Files have been save to "
                         f"{save_path}")

    def plot_test_acc_loss(self,
                           experiment_info_list: Union[List[dict], dict],
                           show: bool = True,
                           save: bool = False,
                           save_path_root: str = None,
                           acc_fig_title: str = 'Accuracy of experiments',
                           loss_fig_title: str = 'Loss of experiments'
                           ):
        """
        Plot acc and loss curve for single or multiple experiments.
        :param loss_fig_title:
        :param acc_fig_title:
        :param experiment_info_list: (List[dict] or dict)  info from different experiments or single experiment info.
        :param show: (bool)  run plt.show()
        :param save: (bool)  run plt.save(), need to specify a save_path_root if true.
        :param save_path_root:
        :return:
        """
        self.logger.info('Plotting test accuracy and loss on server.')

        if save and (save_path_root is None):
            self.logger.error('Save file is set to True but got no save_path_root.')
            raise TypeError

        # clean directory
        if save:
            create_directory(save_path_root)
            abs_path_root = os.path.abspath(save_path_root)
            save_path_root = abs_path_root

            create_directory(save_path_root + f'/test_acc_loss')
            delete_all_files(save_path_root + f'/test_acc_loss')

        experiment_names = []
        experiment_acc = []
        experiment_loss = []

        # if input single experiment info
        if isinstance(experiment_info_list, dict):
            experiment_info_list = [experiment_info_list]

        # Gather server info from all experiments.
        for experiment_info in experiment_info_list:
            experiment_name = experiment_info['name']
            server_info = experiment_info['server_info']
            if not isinstance(server_info, dict):
                self.logger.error(f'Unknown type server info, expected dict, but got {server_info.__class__}.')
                raise TypeError

            if ('test_acc' not in server_info.keys()) or ('test_loss' not in server_info.keys()):
                self.logger.warning(f"Server ({server_info['class']}) do not have 'test_acc' and 'test_loss' "
                                    f"in server info. Skipping the server.")

            else:
                experiment_names.append(experiment_name)
                experiment_acc.append(server_info['test_acc'])
                experiment_loss.append(server_info['test_loss'])

        # plot acc
        plt.figure()
        plt.title(acc_fig_title)
        plt.xlabel('Round')
        plt.ylabel('Accuracy(%)')
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=1)
        for idx in range(len(experiment_names)):
            round_x_label = np.arange(len(experiment_acc[idx])) + 1
            plt.plot(round_x_label, np.array(experiment_acc[idx]) * 100, label=experiment_names[idx])

        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

        if save:
            save_path_fig = save_path_root + f'/test_acc_loss/acc.png'
            plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        # plot loss
        plt.figure()
        plt.title(loss_fig_title)
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=1)
        for idx in range(len(experiment_names)):
            round_x_label = np.arange(len(experiment_loss[idx])) + 1
            plt.plot(round_x_label, experiment_loss[idx], label=experiment_names[idx])

        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

        if save:
            save_path_fig = save_path_root + f'/test_acc_loss/loss.png'
            plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        save_path = os.path.abspath(save_path_root + f'/test_acc_loss')
        self.logger.info(f"Loss and accuracy plotted. Files have been save to "
                         f"{save_path}")

    def generate_abstract(self, experiment_info: dict, save_path: str):
        self.logger.info('Generating experiment abstract.')

        write_lines = []

        # global info
        global_abstract = 'Experiment Global = { \n'
        for key in sorted(experiment_info.keys()):
            if key != 'server_info' and key != 'client_info':
                line = '    ' + str(key) + ' = ' + str(experiment_info[key]).replace('\n', '') + ',\n'
                global_abstract += line
        global_abstract += '}\n\n'
        write_lines.append(deepcopy(global_abstract))

        # server info
        server_info = experiment_info['server_info']
        server_abstract = 'Server = { \n'
        for key in sorted(server_info.keys()):
            line = '    ' + str(key) + ' = ' + str(server_info[key]).replace('\n', '') + ',\n'
            server_abstract += line
        server_abstract += '}\n\n'
        write_lines.append(deepcopy(server_abstract))

        # client info
        for client_info in experiment_info['client_info']:
            client_abstract = f"Client{client_info['client_id']}" + " = { \n"
            for key in sorted(client_info.keys()):
                line = '    ' + str(key) + ' = ' + str(client_info[key]).replace('\n', '') + ',\n'
                client_abstract += line
            client_abstract += '}\n\n'
            write_lines.append(deepcopy(client_abstract))

        with open(save_path, 'w') as f:
            for str_info in write_lines:
                f.write(str_info)

        save_path = os.path.abspath(save_path)
        self.logger.info(f'Experiment abstract generated. Files have been save to {save_path}')

    def plot_train_acc_loss(self,
                            experiment_info: dict,
                            show: bool = True,
                            save: bool = False,
                            save_path_root: str = None,
                            rounds_in_figure: int = 10
                            ):
        self.logger.info('Plotting train accuracy and loss on clients. This may take some time.')

        if save and (save_path_root is None):
            self.logger.error('Save file is set to True but got no save_path_root.')
            raise TypeError

        # clean directory
        if save:
            create_directory(save_path_root)
            abs_path_root = os.path.abspath(save_path_root)
            save_path_root = abs_path_root

            create_directory(save_path_root + f'/train_acc_loss')
            delete_all_files(save_path_root + f'/train_acc_loss')

        for client_info in experiment_info['client_info']:

            client_id = client_info['client_id']
            train_acc = client_info['train_acc']
            train_loss = client_info['train_loss']
            total_rounds = len(train_acc)

            if total_rounds == 0:
                self.logger.warning(f'Total rounds should be > 0. '
                                    f'Client {client_id} have {total_rounds} rounds in total')
                continue
            epoch_nums = len(train_acc[0])
            create_directory(save_path_root + f"/train_acc_loss/client{client_id}")

            figure_nums = math.ceil(total_rounds / rounds_in_figure)
            empty_data_num = (rounds_in_figure - total_rounds % rounds_in_figure) % rounds_in_figure
            for _ in range(empty_data_num):
                train_acc.append(np.array([np.nan] * epoch_nums))
                train_loss.append(np.array([np.nan] * epoch_nums))

            train_loss = np.array(train_loss)
            train_acc = np.array(train_acc)
            for idx in range(figure_nums):
                step = 1 / epoch_nums
                start = idx * rounds_in_figure  # start round (-1)
                end = (idx + 1) * rounds_in_figure  # end round (-1)

                plot_x = np.concatenate([np.arange(i, i+1, step) for i in range(start + 1, end+1)])
                plot_acc = np.concatenate(train_acc[start:end])
                plot_loss = np.concatenate(train_loss[start:end])

                plt.figure()
                plt.plot(plot_x, plot_acc * 100)
                plt.title(f'Train accuracy in client {client_id}')
                plt.xlabel('Round')
                plt.ylabel('Accuracy(%)')
                plt.xticks(np.arange(start + 1, end + 1))
                plt.xlim((start + 1, end + 1))
                plt.grid(True, linestyle='-', alpha=0.4, linewidth=1, axis='x', color='k')
                plt.grid(True, linestyle='--', alpha=0.3, linewidth=1, axis='y')
                if save:
                    save_path_fig = save_path_root + f"/train_acc_loss/client{client_id}/" \
                                                     f"acc_round_{start+1}_{end}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

                plt.figure()
                plt.plot(plot_x, plot_loss)
                plt.title(f'Train loss in client {client_id}')
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.xticks(np.arange(start + 1, end + 1))
                plt.xlim((start + 1, end + 1))
                plt.grid(True, linestyle='-', alpha=0.4, linewidth=1, axis='x', color='k')
                plt.grid(True, linestyle='--', alpha=0.3, linewidth=1, axis='y')
                if save:
                    save_path_fig = save_path_root + f"/train_acc_loss/client{client_id}/" \
                                                     f"loss_round_{start + 1}_{end}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

        plot_path = os.path.abspath(save_path_root + '/train_acc_loss')
        self.logger.info('Train accuracy and loss plotted on clients. '
                         f'Files have been save to {plot_path}')

    def plot_byzantine_defense(self,
                               experiment_info: dict,
                               show: bool = True,
                               save: bool = False,
                               save_path_root: str = None,
                               ):
        pca_2d: bool = True
        tsne_2d: bool = True
        tsne_3d: bool = True
        server_info = experiment_info['server_info']
        if 'attacker_idx' not in experiment_info.keys():
            self.logger.warning("No attacker_idx found in experiment_info, check if you forget to record this or "
                                "this is not a byzantine experiment. Skipping.")
            return

        if 'sampled_attacker_idx' not in server_info.keys():
            self.logger.warning("No sampled_attacker_idx found in server_info"
                                ", check if you forget to record this or "
                                "this is not a byzantine defense experiment. Skipping.")
            return

        if 'tsne_2d_plot_data' not in server_info.keys() or len(server_info['tsne_2d_plot_data']) == 0:
            self.logger.info("No tsne_2d_plot_data found in server_info, skipping tsne_2d."
                             )
            tsne_2d = False

        if 'tsne_3d_plot_data' not in server_info.keys() or len(server_info['tsne_3d_plot_data']) == 0:
            self.logger.info("No tsne_3d_plot_data found in server_info, skipping tsne_3d.")
            tsne_3d = False

        if 'pca_plot_data' not in server_info.keys() or len(server_info['pca_plot_data']) == 0:
            self.logger.info("No pca_plot_data found in server_info, skipping pca_2d.")
            pca_2d = False

        self.logger.info('Plotting byzantine defense situation.')
        if save and (save_path_root is None):
            self.logger.error('Save file is set to True but got no save_path_root.')
            raise TypeError

        # clean directory
        if save:
            create_directory(save_path_root)
            abs_path_root = os.path.abspath(save_path_root)
            save_path_root = abs_path_root

            create_directory(save_path_root + f'/byzantine_defense')
            delete_all_files(save_path_root + f'/byzantine_defense')

        real_attacker_idx = experiment_info['attacker_idx']
        server_sampled_attacker_idx = server_info['sampled_attacker_idx']
        assert len(server_sampled_attacker_idx) == experiment_info['rounds']

        manslaughter_amount_list = []
        missed_amount_list = []
        right_amount_list = []

        for sampled_attacker_idx in server_sampled_attacker_idx:
            killed_set = set(sampled_attacker_idx)
            real_set = set(real_attacker_idx)

            manslaughter_amount_list.append(len(killed_set - real_set))
            missed_amount_list.append(len(real_set - killed_set))
            right_amount_list.append(len(killed_set & real_set))

        plot_x = np.arange(1, experiment_info['rounds'] + 1)
        plt.figure()
        plt.plot(plot_x, manslaughter_amount_list, label='Manslaughter')
        plt.plot(plot_x, missed_amount_list, label='Missed')
        plt.plot(plot_x, right_amount_list, label='Correct Killed')
        plt.xticks(np.arange(1, experiment_info['rounds'] + 1))
        plt.xlim((0, experiment_info['rounds'] + 1))
        plt.title(f'Byzantine killed attackers')
        plt.xlabel('Round')
        plt.ylabel('Amount')
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.grid(True, linestyle='--', alpha=0.3, linewidth=1)
        if save:
            save_path_fig = save_path_root + f"/byzantine_defense/byzantine_kills.png"
            plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        if pca_2d:
            create_directory(save_path_root + f'/byzantine_defense/pca_visualize_2D')
            attacker_idx = experiment_info['attacker_idx']
            for round_idx in range(experiment_info['rounds']):
                if round_idx == 0:
                    continue

                plot_client_idx = np.array(server_info['pca_plot_client_idx'][round_idx])
                pca_plot_data = np.array(server_info['pca_plot_data'][round_idx])
                sampled_attacker_idx = np.array(server_info['sampled_attacker_idx'][round_idx])

                attacker_idx = [idx for idx, val in enumerate(plot_client_idx) if val in attacker_idx]
                sampled_attacker_idx = [idx for idx, val in enumerate(plot_client_idx) if val in sampled_attacker_idx]
                assert len(pca_plot_data.shape) == 2
                plt.figure()
                plt.xlabel('PCA X')
                plt.ylabel('PCA Y')
                plt.scatter(pca_plot_data[:, 0], pca_plot_data[:, 1], c='grey')
                plt.scatter(pca_plot_data[attacker_idx, 0],
                            pca_plot_data[attacker_idx, 1],
                            c='red', label='Real Attacker')
                if len(sampled_attacker_idx) > 0:
                    plt.scatter(pca_plot_data[sampled_attacker_idx, 0],
                                pca_plot_data[sampled_attacker_idx, 1],
                                c='black', marker='x', label='Sampled Attacker')

                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.title('Robust Sampler Result (PCA)')
                if save:
                    save_path_fig = save_path_root + f"/byzantine_defense/pca_visualize_2D/round_{round_idx + 1}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()
        if tsne_2d:
            create_directory(save_path_root + f'/byzantine_defense/tsne_visualize_2D')
            attacker_idx = experiment_info['attacker_idx']
            for round_idx in range(experiment_info['rounds']):
                if round_idx == 0:
                    continue

                plot_client_idx = np.array(server_info['pca_plot_client_idx'][round_idx])
                tsne_plot_data = np.array(server_info['tsne_2d_plot_data'][round_idx])
                sampled_attacker_idx = np.array(server_info['sampled_attacker_idx'][round_idx])

                sorted_plot_data = tsne_plot_data[plot_client_idx]
                assert len(sorted_plot_data.shape) == 2
                plt.figure()
                plt.xlabel('TSNE X')
                plt.ylabel('TSNE Y')
                plt.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1], c='grey')
                plt.scatter(sorted_plot_data[attacker_idx, 0],
                            sorted_plot_data[attacker_idx, 1],
                            c='red', label='Real Attacker')
                if len(sampled_attacker_idx) > 0:
                    plt.scatter(sorted_plot_data[sampled_attacker_idx, 0],
                                sorted_plot_data[sampled_attacker_idx, 1],
                                c='black', marker='x', label='Sampled Attacker')

                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.title('Robust Sampler Result (TSNE-2D)')
                if save:
                    save_path_fig = save_path_root + f"/byzantine_defense/tsne_visualize_2D/round_{round_idx + 1}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

        if tsne_3d:
            create_directory(save_path_root + f'/byzantine_defense/tsne_visualize_3D')
            attacker_idx = experiment_info['attacker_idx']
            for round_idx in range(experiment_info['rounds']):
                if round_idx == 0:
                    continue

                plot_client_idx = np.array(server_info['pca_plot_client_idx'][round_idx])
                tsne_plot_data = np.array(server_info['tsne_3d_plot_data'][round_idx])
                sampled_attacker_idx = np.array(server_info['sampled_attacker_idx'][round_idx])

                sorted_plot_data = tsne_plot_data[plot_client_idx]
                assert len(sorted_plot_data.shape) == 2 and sorted_plot_data.shape[1] == 3
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.set_xlabel('TSNE X')
                ax.set_ylabel('TSNE Y')
                ax.set_zlabel('TSNE Z')
                ax.set_title('Robust Sampler Result (TSNE-3D)')
                benign_plot_data = sorted_plot_data[~np.isin(np.arange(len(sorted_plot_data)), attacker_idx)]

                ax.scatter(benign_plot_data[:, 0],
                           benign_plot_data[:, 1],
                           benign_plot_data[:, 2],
                           c='grey', marker='o')
                ax.scatter(sorted_plot_data[attacker_idx, 0],
                           sorted_plot_data[attacker_idx, 1],
                           sorted_plot_data[attacker_idx, 2],
                            c='red', label='Real Attacker')
                if len(sampled_attacker_idx) > 0:
                    ax.scatter(sorted_plot_data[sampled_attacker_idx, 0],
                               sorted_plot_data[sampled_attacker_idx, 1],
                               sorted_plot_data[sampled_attacker_idx, 2],
                                c='black', marker='x', label='Sampled Attacker')

                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.title('Robust Sampler Result (TSNE-3D)')
                if save:
                    save_path_fig = save_path_root + f"/byzantine_defense/tsne_visualize_3D/round_{round_idx + 1}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

        plot_path = os.path.abspath(save_path_root + '/byzantine_defense')
        self.logger.info('Byzantine defense plotted. '
                         f'Files have been save to {plot_path}')

    def plot_cluster(self,
                     experiment_info: dict,
                     show: bool = True,
                     save: bool = False,
                     save_path_root: str = None,
                     ):
        pca_2d: bool = True
        tsne_2d: bool = True
        tsne_3d: bool = True
        server_info = experiment_info['server_info']
        if 'attacker_idx' not in experiment_info.keys():
            self.logger.warning("No attacker_idx found in experiment_info, check if you forget to record this or "
                                "this is not a byzantine experiment. Skipping.")
            return

        if 'sampled_attacker_idx' not in server_info.keys():
            self.logger.warning("No sampled_attacker_idx found in server_info"
                                ", check if you forget to record this or "
                                "this is not a byzantine defense experiment. Skipping.")
            return

        if 'cluster_byzantine_robust' not in server_info.keys():
            self.logger.warning("No cluster_byzantine_robust found in server_info"
                                ", check if you forget to record this or "
                                "this is not a byzantine defense with cluster algorithm experiment. Skipping.")
            return

        if 'tsne_2d_plot_data' not in server_info.keys() or len(server_info['tsne_2d_plot_data']) == 0:
            self.logger.info("No tsne_2d_plot_data found in server_info, skipping tsne_2d."
                             )
            tsne_2d = False

        if 'tsne_3d_plot_data' not in server_info.keys() or len(server_info['tsne_3d_plot_data']) == 0:
            self.logger.info("No tsne_3d_plot_data found in server_info, skipping tsne_3d.")
            tsne_3d = False

        if 'pca_plot_data' not in server_info.keys() or len(server_info['pca_plot_data']) == 0:
            self.logger.info("No pca_plot_data found in server_info, skipping pca_2d.")
            pca_2d = False

        self.logger.info('Plotting cluster decision. This may take some time.')

        if save and (save_path_root is None):
            self.logger.error('Save file is set to True but got no save_path_root.')
            raise TypeError

        # clean directory
        if save:
            create_directory(save_path_root)
            abs_path_root = os.path.abspath(save_path_root)
            save_path_root = abs_path_root

            create_directory(save_path_root + f'/byzantine_defense_cluster_decision')
            delete_all_files(save_path_root + f'/byzantine_defense_cluster_decision')

        assert len(server_info['pca_plot_data']) == \
               len(server_info['pca_plot_client_idx']) == \
               len(server_info['cluster_result']) == \
               experiment_info['rounds']

        if pca_2d:
            create_directory(save_path_root + f'/byzantine_defense_cluster_decision/pca_visualize')
            attacker_idx = experiment_info['attacker_idx']

            for round_idx in range(experiment_info['rounds']):
                if round_idx == 0:
                    continue

                cluster_result = np.array(server_info['cluster_result'][round_idx])
                cluster_pca_plot_client_idx = np.array(server_info['pca_plot_client_idx'][round_idx])
                cluster_pca_plot_data = np.array(server_info['pca_plot_data'][round_idx])
                sampled_attacker_idx = np.array(server_info['sampled_attacker_idx'][round_idx])

                sorted_plot_data = cluster_pca_plot_data[cluster_pca_plot_client_idx]
                assert len(sorted_plot_data.shape) == 2

                plt.figure()
                plt.xlabel('PCA X')
                plt.ylabel('PCA Y')
                cluster_color_map = plt.colormaps.get_cmap('Set3')
                plt.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1], cmap=cluster_color_map, c=cluster_result)
                plt.scatter(sorted_plot_data[attacker_idx, 0],
                            sorted_plot_data[attacker_idx, 1],
                            facecolors='none', edgecolors='red', linewidths=1, label='Real Attacker')

                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.title('Cluster Result')
                if save:
                    save_path_fig = save_path_root + f"/byzantine_defense_cluster_decision/pca_visualize/" \
                                                     f"round_{round_idx+1}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

        if tsne_2d:
            create_directory(save_path_root + f'/byzantine_defense_cluster_decision/tsne_visualize_2D')
            attacker_idx = experiment_info['attacker_idx']

            for round_idx in range(experiment_info['rounds']):
                if round_idx == 0:
                    continue

                cluster_result = np.array(server_info['cluster_result'][round_idx])
                cluster_pca_plot_client_idx = np.array(server_info['pca_plot_client_idx'][round_idx])
                cluster_pca_plot_data = np.array(server_info['tsne_2d_plot_data'][round_idx])
                sampled_attacker_idx = np.array(server_info['sampled_attacker_idx'][round_idx])

                sorted_plot_data = cluster_pca_plot_data[cluster_pca_plot_client_idx]
                assert len(sorted_plot_data.shape) == 2

                plt.figure()
                plt.xlabel('TSNE X')
                plt.ylabel('TSNE Y')
                cluster_color_map = plt.colormaps.get_cmap('Set3')
                plt.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1], cmap=cluster_color_map, c=cluster_result)
                plt.scatter(sorted_plot_data[attacker_idx, 0],
                            sorted_plot_data[attacker_idx, 1],
                            facecolors='none', edgecolors='red', linewidths=1, label='Real Attacker')

                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.title('Cluster Result')
                if save:
                    save_path_fig = save_path_root + f"/byzantine_defense_cluster_decision/tsne_visualize_2D/" \
                                                     f"round_{round_idx + 1}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

        if tsne_3d:
            create_directory(save_path_root + f'/byzantine_defense_cluster_decision/tsne_visualize_3D')
            attacker_idx = experiment_info['attacker_idx']
            for round_idx in range(experiment_info['rounds']):
                if round_idx == 0:
                    continue

                plot_client_idx = np.array(server_info['pca_plot_client_idx'][round_idx])
                tsne_plot_data = np.array(server_info['tsne_3d_plot_data'][round_idx])
                sampled_attacker_idx = np.array(server_info['sampled_attacker_idx'][round_idx])

                sorted_plot_data = tsne_plot_data[plot_client_idx]
                assert len(sorted_plot_data.shape) == 2 and sorted_plot_data.shape[1] == 3
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.set_xlabel('TSNE X')
                ax.set_ylabel('TSNE Y')
                ax.set_zlabel('TSNE Z')
                cluster_color_map = plt.colormaps.get_cmap('Set3')
                ax.set_title('Robust Sampler Result (TSNE-3D)')

                ax.scatter(sorted_plot_data[:, 0],
                           sorted_plot_data[:, 1],
                           sorted_plot_data[:, 2],
                           cmap=cluster_color_map, c=cluster_result)
                ax.scatter(sorted_plot_data[attacker_idx, 0],
                           sorted_plot_data[attacker_idx, 1],
                           sorted_plot_data[attacker_idx, 2],
                           facecolors='none', edgecolors='red', linewidths=1, label='Real Attacker')

                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.title('Robust Sampler Result (TSNE-3D)')
                if save:
                    save_path_fig = save_path_root + f"/byzantine_defense_cluster_decision/tsne_visualize_3D/" \
                                                     f"round_{round_idx + 1}.png"
                    plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

                if show:
                    plt.show()
                plt.close()

        plot_path = os.path.abspath(save_path_root + '/byzantine_defense_cluster_decision')
        self.logger.info('Cluster decision plotted. '
                         f'Files have been save to {plot_path}')

    def plot_asr(self,
                 experiment_info: dict,
                 show: bool = True,
                 save: bool = False,
                 save_path_root: str = None,
                 ):
        server_info = experiment_info['server_info']
        if 'backdoor_asr' not in server_info.keys() or len(server_info['backdoor_asr']) == 0:
            self.logger.warning("No backdoor_asr found in server_info, check if you forget to record this or "
                                "this is not a backdoor experiment. Skipping.")
            return

        if 'backdoor' not in server_info.keys() or not server_info['backdoor']:
            self.logger.warning("No backdoor found in server_info, check if you forget to record this or "
                                "this is not a backdoor experiment. Skipping.")
            return

        self.logger.info('Plotting ASR.')

        if save and (save_path_root is None):
            self.logger.error('Save file is set to True but got no save_path_root.')
            raise TypeError

        # clean directory
        if save:
            create_directory(save_path_root)
            abs_path_root = os.path.abspath(save_path_root)
            save_path_root = abs_path_root

            create_directory(save_path_root + f'/backdoor_asr')
            delete_all_files(save_path_root + f'/backdoor_asr')

        plot_y = np.array(server_info['backdoor_asr']) * 100
        plot_x = np.arange(2, len(plot_y) + 2)  # Asr calculated from round 2
        plt.figure()
        plt.xlabel('Round')
        plt.ylabel('ASR(%)')
        plt.plot(plot_x, plot_y)

        plt.title('Backdoor ASR')
        if save:
            save_path_fig = save_path_root + f"/backdoor_asr/asr.png"
            plt.savefig(save_path_fig, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        plot_path = os.path.abspath(save_path_root + '/backdoor_asr')
        self.logger.info('Backdoor asr plotted. '
                         f'Files have been save to {plot_path}')











