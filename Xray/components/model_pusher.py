import os
import sys

from Xray.exception import XRayException
from Xray.logger import logging
from Xray.entity.artifact_entity import ModelPusherArtifact
from Xray.entity.config_entity import ModelPusherConfig

class ModelPusher:

    def __init__(self, model_pusher_config:ModelPusherConfig):

        self.model_pusher_config = model_pusher_config

    def build_and_push_bento_image(self):

        logging.info("Entered build_and_push_bento_image Method")

        try:
            logging.info("Building The bento From bentofile")


            os.system("bentoml build")

            logging.info("Creating Docker Image For Bento")

            logging.info("Logging Into ECR")

            os.system(
            "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 570934197674.dkr.ecr.us-east-1.amazonaws.com"
                )

            logging.info("Logged into ECR")

            logging.info("Pushing bento image to ECR")

            os.system(
                    "docker push 570934197674.dkr.ecr.us-east-1.amazonaws.com/musab:latest"
                )

            logging.info("Pushed bento image to ECR")

            logging.info(
                "Exited build_and_push_bento_image method of ModelPusher class"
            )

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_pusher(self)-> ModelPusherArtifact:

        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """


        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            self.build_and_push_bento_image()

            model_pusher_artifact= ModelPusherArtifact(
                bentoml_model_name = self.model_pusher_config.bentoml_model_name,
                bentoml_service_name = self.model_pusher_config.bentoml_service_name
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)

