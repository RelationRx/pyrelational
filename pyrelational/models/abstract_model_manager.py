import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from torch.utils.data import DataLoader


class ModelManager(ABC):
    def __init__(
        self,
        model_class: Type[Any],
        model_config: Union[str, Dict[str, Any]],
        trainer_config: Union[str, Dict[str, Any]],
    ):
        """
        Abstract class used to wrap models to interact with the ActiveLearningStrategy modules.
        It handles model instantiation at each iteration, training, testing, and queries.

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

    def init_trainer(self, trainer_config: Dict[str, Any]) -> Any:
        """
        Initialise trainer.

        :param trainer_config: a dictionary containing the config required to instantiate the trainer module/function
        :return: trainer module/function
        """
        pass

    def init_model(self) -> Any:
        """
        Initialise model instance(s).

        :return: an instance of self.model_class based on self.model_config
        """
        return self.model_class(**self.model_config)

    @abstractmethod
    def train(self, train_loader: DataLoader[Any], valid_loader: Optional[DataLoader[Any]] = None) -> None:
        """
        Run train routine.

        :param train_loader: pytorch dataloader for training set
        :param valid_loader: pytorch dataloader for validation set
        """
        pass

    @abstractmethod
    def test(self, loader: DataLoader[Any]) -> Dict[str, float]:
        """
        Run test routine.

        :param loader: pytorch dataloader for test set
        :return: performance metrics
        """
        pass

    def __call__(self, loader: DataLoader[Any]) -> Any:
        """
        Compute predictions for each sample in dataloader.

        :param loader: pytorch dataloader
        :return: predictions
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_class.__name__})"
