import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

from torch.utils.data import DataLoader


class ModelManager(ABC):
    """
    Abstract class used to wrap models to interact with the ActiveLearningStrategy modules.
    It handles model instantiation at each iteration, training, testing, and queries.
    """

    def __init__(
        self,
        model_class: Type[Any],
        model_config: Union[str, Dict],
        trainer_config: Union[str, Dict],
    ):
        """
        :param model_class: a model constructor (e.g. torch.nn.Linear)
        :param model_config: a dictionary containing the config required to instantiate a model form the model_class
                (e.g. {in_features=100, out_features=34, bias=True, device=None, dtype=None} for a torch.nn.Linear
                constructor)
        :param trainer_config: a dictionary containing the config required to instantiate the trainer module/function
        """
        super(ModelManager, self).__init__()

        self.model_class = model_class
        self.model_config = json.load(open(model_config, "r")) if isinstance(model_config, str) else model_config
        self.current_model = None
        self.trainer_config = (
            json.load(open(trainer_config, "r")) if isinstance(trainer_config, str) else trainer_config
        )

    def init_trainer(self, trainer_config: Dict) -> Any:
        """
        Initialise trainer which trains model.

        :param trainer_config: a dictionary containing the config required to instantiate the trainer module/function
        :return: trainer module/function
        """
        pass

    def init_model(self) -> Any:
        """
        Initialise an instance of the user defined model.

        :return: an instance of self.model_class based on self.model_config
        """
        return self.model_class(**self.model_config)

    @abstractmethod
    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None) -> None:
        """
        Train model.

        :param train_loader: pytorch dataloader for training set
        :param valid_loader: pytorch dataloader for validation set
        :return: none
        """
        pass

    @abstractmethod
    def test(self, loader: DataLoader) -> Dict:
        """
        Test model.

        :param loader: pytorch dataloader for test set
        :return: performance metrics
        """
        pass

    def __call__(self, loader: DataLoader) -> Any:
        """
        Call method to output model predictions

        :param loader: pytorch dataloader
        :return: model predictions for each sample in dataloader
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_class.__name__})"
