import sys

from Xray.cloud_storage.s3_operation import S3Operation
from Xray.constants.training_pipeline import *
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.entity.config_entity import DataIngestionConfig

from Xray.exception import XRayException
from Xray.logger import logging


class DataIngestion:

    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

        self.s3 = S3Operation()



    def get_data_from_s3(self):
        try:
            logging.info("Entered The get_data_from_s3 Method Of Data Ingestion Class")

            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                bucket_folder_name=self.data_ingestion_config.S3_data_folder
            )

            logging.info("Exited The get_data_from_s3 Method Of Data Ingestion Class")
        except Exception as e:
            raise XRayException(e, sys)
        
    
    def initiate_data_ingestion(self):
        try:
            self.get_data_from_s3()

            data_ingestion_artifact:DataIngestionArtifact=DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path, test_file_path=self.data_ingestion_config.test_data_path
            )

            logging.info(
                "Exited The initiate_data_imgs Function Of Data Ingestion Class"
            )

            return data_ingestion_artifact
        
        except Exception as e:
            raise XRayException(e, sys)
        