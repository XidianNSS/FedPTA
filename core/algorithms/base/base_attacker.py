from fedsimu.core.algorithms.base.base_client import BaseClient
from abc import abstractmethod, ABC


class BaseAttacker(BaseClient):
    """
    This class attempts to implement a base class for Byzantine clients.
    Multiple inheritance should be used when trying to implement a Byzantine client on a particular algorithm.
    Example:

        class MyAttacker(MyAttacker, FedAvgClient):  # This order cannot be reversed
            def __init__(self, ...):
                FedAvgClient.init(self, ...)  # super.__init__ order cannot be reversed
                MyAttacker.init(self, ...)

    Attack process usually is:
        before_train(global_info)  # benign

        attack_before_train(global_info)  # byzantine

        train(global_info)  # benign

        attack_after_train(global_info)  # byzantine

        after_train(global_info)  # benign

    Attention:
        Usually attack process will start after first round(usually starts in round 2). Cause before round 1 server
        send no data to clients.
    """
    def __init__(self):
        self.local_records['attacker'] = True

    def local_process(self, global_info: dict):
        # log info.
        self.logger.info("[Round {}]Start training on client {}(Attacker)."
                         .format(global_info['round'] + 1,
                                 self.client_id))
        self.log_client_abstract()

        # refresh package
        self.uplink_package = {'client_id': self.client_id, 'round': global_info['round'] + 1, 'attacker': True}

        self.before_train(global_info)  # Usually load global info from server.
        self.attack_before_train(global_info)  # byzantine
        self.train(global_info)  # Usually train process
        self.attack_after_train(global_info)  # byzantine
        self.after_train(global_info)  # Usually upload local info to server.
        self.record_model(global_info)

        return self.uplink_package

    @abstractmethod
    def attack_before_train(self, global_info: dict):
        raise NotImplementedError

    @abstractmethod
    def attack_after_train(self, global_info: dict):
        raise NotImplementedError


