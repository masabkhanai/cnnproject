import sys
from typing import Tuple

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset

from Xray.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact

from Xray.entity.config_entity import ModelEvaluationConfig

from Xray.logger import logging
from Xray.ml.model.arch import Net
from Xray.exception import XRayException


class ModelEvaluation:

    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_artifact: ModelTrainerArtifact, model_evaluation_config: ModelEvaluationConfig,):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config

    def configuration(self) -> Tuple[DataLoader, Module, Module]:

        logging.info("Entered The Configuration Method")

        try:

            test_dataloader:DataLoader = (self.data_transformation_artifact.transformed_test_object)
            model:Module = Net()

            model.load_state_dict(torch.load(self.model_trainer_artifact.trained_model_path, map_location=self.model_evaluation_config.device))

            model.to(self.model_evaluation_config.device)

            cost: Module = CrossEntropyLoss()

            # optimizer: Optimizer = SGD(model.parameters(), **self.model_evaluation_config.optimizer_params)

            model.eval()

            logging.info("Exited The Configurattion Method")

            return test_dataloader, model, cost

        except Exception as e:
            raise XRayException(e, sys)

    def test_net(self)-> float:
        logging.info("Entered The test_net Method")

        try:
            test_dataloader, net, cost = self.configuration()

            with torch.no_grad():
                holder = []

                for _, data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evaluation_config.device)

                    labels = data[1].to(self.model_evaluation_config.device)

                    output = net(images)

                    loss = cost(output, labels)

                    predictions = torch.argmax(output, 1)

                    for i in zip(images, labels, predictions):
                        h = list(i)

                        holder.append(h)

                    logging.info(f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}")

                    self.model_evaluation_config.test_accuracy+=(predictions==labels).sum().item()

                    self.model_evaluation_config.total_batch+=1

                    self.model_evaluation_config.total +=labels.size(0)

                    logging.info(
                        f"Model  -->   Loss : {self.model_evaluation_config.test_loss/ self.model_evaluation_config.total_batch} Accuracy : {(self.model_evaluation_config.test_accuracy / self.model_evaluation_config.total) * 100} %"
                    )
                    accuracy = (self.model_evaluation_config.test_accuracy / self.model_evaluation_config.total) *100

                    logging.info("Exited the test_net method of Model evaluation class")

            return accuracy

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        logging.info(
            "Entered the initiate_model_evaluation method of Model evaluation class"
        )

        try:
            accuracy = self.test_net()

            model_evaluation_artifact : ModelEvaluationArtifact = (
                ModelEvaluationArtifact(model_accuracy = accuracy)
            )

            logging.info(
                "Exited the initiate_model_evaluation method of Model evaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)