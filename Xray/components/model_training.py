import os
import torch
import torchvision
from Xray.exception import XRayException
from Xray.components.data_transformation import DataTransformationArtifact
import bentoml
import joblib
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm
from Xray.entity.config_entity import ModelTrainerConfig
from Xray.constants.training_pipeline import *
from Xray.ml.model.arch import Net
from Xray.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from Xray.logger import logging
import sys
from torch.nn import Module



class ModelTrainer:

    def __init__(self, data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig,):

        self.model_trainer_config:ModelTrainerConfig=model_trainer_config

        self.data_transformation_artifact:DataTransformationArtifact = data_transformation_artifact

        self.model:Module=Net()

    def train(self, optimizer:Optimizer)-> None:

        # Description: To Train The Model
        # Input: Model, Device, train_loader, optimizer, epochs
        # output: Loss, batch_id and accuracy
        
        logging.info("Entered The Train Method Of Model Trainer Class")

        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct :int = 0
            processed = 0

            for batch_idx, (data, target) in enumerate(pbar):

                data, target = data.to(DEVICE), target.to(DEVICE)

                #initiaization of gradient
                optimizer.zero_grad()

                y_pred = self.model(data)

                #calculating the loss
                weights = torch.tensor([1.5, 1.0]).to(DEVICE)

                loss = F.nll_loss(
                    y_pred,
                    target,
                    weight=weights
                )

                #Backprop
                loss.backward()
                optimizer.step()

                pred = y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                processed+=len(data)

                pbar.set_description(desc=f"Loss={loss.item()} Batch Id = {batch_idx} Accuracy = {100*correct/processed:0.2f}")

            logging.info("Exited The Train Method")
        except Exception as e:
            raise XRayException(e, sys)

    def test(self) -> None:

        try:

            logging.info("Entered The Test Method Of Model Trainer Class")

            self.model.eval()

            correct = 0
            test_loss: float = 0.0

            # ADD THIS
            pred_counts = {
                0: 0,
                1: 0
            }

            actual_counts = {
                0: 0,
                1: 0
            }


            with torch.no_grad():

                for (data, target) in self.data_transformation_artifact.transformed_test_object:

                    data, target = data.to(DEVICE), target.to(DEVICE)

                    output = self.model(data)


                    pred = output.argmax(dim=1, keepdim=True)


                    # prediction count
                    pred_counts[0] += (pred == 0).sum().item()
                    pred_counts[1] += (pred == 1).sum().item()


                    # actual label count
                    actual_counts[0] += (target == 0).sum().item()
                    actual_counts[1] += (target == 1).sum().item()


                    test_loss += F.nll_loss(
                        output,
                        target,
                        reduction="sum"
                    ).item()


                    correct += pred.eq(
                        target.view_as(pred)
                    ).sum().item()



            test_loss /= len(
                self.data_transformation_artifact.transformed_test_object.dataset
            )


            print("\nActual Dataset Count:")
            print(actual_counts)


            print("\nModel Prediction Count:")
            print(pred_counts)


            print(
                "\nTest Set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                    test_loss,
                    correct,
                    len(self.data_transformation_artifact.transformed_test_object.dataset),
                    100.0 * correct / len(self.data_transformation_artifact.transformed_test_object.dataset)
                )
            )


            logging.info("Exited Test Method")


        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            logging.info("Entered initiate_model_trainer Method Of Model Trainer Class")

            model:Module = self.model.to(self.model_trainer_config.device)
            optimizer:Optimizer = torch.optim.SGD(model.parameters(), **self.model_trainer_config.optimizer_params)

            scheduler: _LRScheduler = StepLR(optimizer=optimizer, **self.model_trainer_config.scheduler_params)

            for epoch in range(1, self.model_trainer_config.epochs+1):

                print("Epoch:", epoch)

                self.train(optimizer=optimizer)

                # optimizer.step()
                scheduler.step()
                self.test()

                os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)
            torch.save(model.state_dict(), self.model_trainer_config.trained_model_path)

            train_transforms_obj = joblib.load(self.data_transformation_artifact.train_transform_file_path)

            bentoml.pytorch.save_model(name=self.model_trainer_config.trained_bentoml_model_name,model=model, custom_objects={self.model_trainer_config.train_transforms_key:train_transforms_obj})

            model_trainer_artifact:ModelTrainerArtifact = ModelTrainerArtifact(
                    trained_model_path=self.model_trainer_config.trained_model_path
                )

            logging.info(
                    'Exited The intiate_model_method'
                )
            return model_trainer_artifact

        except Exception as e:
            raise XRayException( e, sys)
            



        

            
        

        