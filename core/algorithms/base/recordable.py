from abc import abstractmethod


class Recordable(object):
    @abstractmethod
    def get_record_abstract(self) -> dict:
        raise NotImplementedError















