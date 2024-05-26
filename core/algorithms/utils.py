import torch.utils.data as Data
import torch
import numpy as np


def serialize_model_gradients(model: torch.nn.Module) -> np.ndarray:
    """_summary_

    Args:
        model (torch.nn.Module): _description_

    Returns:
        torch.Tensor: _description_
    """
    gradients = [param.grad.data.view(-1) for param in model.parameters()]
    m_gradients = torch.cat(gradients)
    m_gradients = m_gradients.cpu().detach().numpy()

    return m_gradients


def deserialize_model_gradients(model: torch.nn.Module, gradients: np.ndarray):
    gradients = torch.tensor(gradients)
    idx = 0
    for parameter in model.parameters():
        layer_size = parameter.grad.numel()
        shape = parameter.grad.shape

        parameter.grad.data[:] = gradients[idx:idx + layer_size].view(shape)[:]
        idx += layer_size


def serialize_model(model: torch.nn.Module) -> np.ndarray:
    """Unfold model parameters

    Unfold every layer of model, concate all of tensors into one.
    Return a `torch.Tensor` with shape (size, ).

    Args:
        model (torch.nn.Module): model to serialize.
    """

    parameters = [param.data.view(-1) for param in model.parameters()]
    m_parameters = torch.cat(parameters)
    m_parameters = m_parameters.cpu().detach().numpy()

    return m_parameters


def deserialize_model(model: torch.nn.Module,
                      serialized_parameters: np.ndarray,
                      mode="copy"):
    """Assigns serialized parameters to model.parameters.
    This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
    NOTE: this function manipulates ``model.parameters``.

    Args:
        model (torch.nn.Module): model to deserialize.
        serialized_parameters (torch.Tensor): serialized model parameters.
        mode (str): deserialize mode. "copy" or "add".
    """

    serialized_parameters = torch.tensor(serialized_parameters)
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        if mode == "copy":
            parameter.data.copy_(
                serialized_parameters[current_index:current_index +
                                                    numel].view(size))
        elif mode == "add":
            parameter.data.add_(
                serialized_parameters[current_index:current_index +
                                                    numel].view(size))
        else:
            raise ValueError(
                "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                .format(mode))
        current_index += numel


def replace_data(origin: np.ndarray, trigger: np.ndarray, x: int, y: int):
    """
    Replace data [x: x + x_delta, y: y + y_delta] with trigger (where np.nan will not be replaced)
    Attention! Use list to index an array may cause error.

    Example:
        a = np.ones([10, 3, 10, 10])
        b = np.ones([10, 3, 10, 10])
        trigger = np.zeros([3, 3])
        replace_data(a[[0, 1, 2, 3], 0], trigger, 0, 0)
        replace_data(b[0:4, 0], trigger, 0, 0)

    Here a will not be replaced but b will be replaced.

    :param origin: np.ndarray  len(shape) == 2 or 3, if 3 we change origin[:]
    :param trigger: np.ndarray  len(shape) == 2
    :param x: start x
    :param y: start y
    :return:
    """
    assert len(origin.shape) == 2 or len(origin.shape) == 3
    assert len(trigger.shape) == 2

    x_end = x + trigger.shape[0]
    y_end = y + trigger.shape[1]
    x_max = 0
    y_max = 0
    if len(origin.shape) == 2:
        x_max = origin.shape[0]
        y_max = origin.shape[1]
    else:
        x_max = origin.shape[1]
        y_max = origin.shape[2]

    nan_mask = np.isnan(trigger)

    assert 0 <= x < x_max
    assert 0 <= x_end < x_max
    assert 0 <= y < y_max
    assert 0 <= y_end < y_max
    if len(origin.shape) == 2:
        trigger_with_data = np.where(nan_mask, origin[x:x_end, y:y_end], trigger)
        origin[x:x_end, y:y_end] = trigger_with_data
    else:
        trigger_with_data = np.where(nan_mask, origin[:, x:x_end, y:y_end], trigger)
        origin[:, x:x_end, y:y_end] = trigger_with_data

