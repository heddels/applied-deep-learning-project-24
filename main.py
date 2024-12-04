#from src.media_bias_detection import logger
from media_bias_detection import logger
from media_bias_detection.pipeline.stage_1_data_processing import InitSubtaskPipeline

STAGE_NAME = "Initialize_Subtasks_Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = InitSubtaskPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e