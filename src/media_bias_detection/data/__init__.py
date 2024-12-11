"""Data initialization and task definitions for MTL model."""

from typing import List
import itertools

from .task import Task, ClassificationSubTask, MultiLabelClassificationSubTask, POSSubTask
from media_bias_detection.utils.logger import general_logger


# initializing the sub-tasks I want to use
st_1_cw_hard_03 = ClassificationSubTask(
task_id=3,
filename="03_CW_HARD/preprocessed.csv",
id=300001)
st_1_me_too_ma_108 = MultiLabelClassificationSubTask(
num_classes=2,
num_labels=2,
task_id=108,
filename="108_MeTooMA/preprocessed.csv",
id=10801,
tgt_cols_list=["hate_speech_label", "sarcasm_label"],
)
st_1_mdgender_116 = ClassificationSubTask(
task_id=116,
id=11601,
filename="116_MDGender/preprocessed.csv",
num_classes=6
)
st_1_mpqa_103 = ClassificationSubTask(
task_id=103,
id=10301,
filename="103_MPQA/preprocessed.csv")
st_1_stereotype_109 = ClassificationSubTask(
task_id=109,
id=10901,
filename="109_stereotype/preprocessed.csv")
st_2_stereotype_109 = MultiLabelClassificationSubTask(
task_id=109,
id=10902,
filename="109_stereotype/preprocessed.csv",
tgt_cols_list=["stereotype_explicit_label", "stereotype_explicit_label"],
num_classes=2,
num_labels=2,
)
st_1_good_news_everyone_42 = POSSubTask(
tgt_cols_list=["cue_pos"],
task_id=42,
id=42001,
filename="42_GoodNewsEveryone/preprocessed.csv"
)
st_2_good_news_everyone_42 = POSSubTask(
tgt_cols_list=["experiencer_pos"],
task_id=42,
id=42002,
filename="42_GoodNewsEveryone/preprocessed.csv",
)
st_1_pheme_12 = ClassificationSubTask(
task_id=12,
id=12001,
filename="12_PHEME/preprocessed.csv")
st_2_pheme_12 = ClassificationSubTask(
task_id=12,
id=12002,
filename="12_PHEME/preprocessed.csv",
tgt_cols_list=["veracity_label"],
num_classes=3,
)
st_1_babe_10 = ClassificationSubTask(
task_id=10,
id=10001,
filename="10_BABE/preprocessed.csv",
num_classes=2)
st_2_babe_10 = POSSubTask(
task_id=10,
id=10002,
filename="10_BABE/preprocessed.csv",
tgt_cols_list=["biased_words"])
st_1_gwsd_128 = ClassificationSubTask(
task_id=128,
num_classes=3,
filename="128_GWSD/preprocessed.csv",
id=12801)

# Tasks
cw_hard_03 = Task(task_id=3, subtasks_list=[st_1_cw_hard_03])
babe_10 = Task(task_id=10, subtasks_list=[st_1_babe_10, st_2_babe_10])
me_too_ma_108 = Task(task_id=108, subtasks_list=[st_1_me_too_ma_108])
mdgender_116 = Task(task_id=116, subtasks_list=[st_1_mdgender_116])
pheme_12 = Task(task_id=12, subtasks_list=[st_2_pheme_12, st_1_pheme_12])
mpqa_103 = Task(task_id=103, subtasks_list=[st_1_mpqa_103])
stereotype_109 = Task(task_id=109, subtasks_list=[st_1_stereotype_109,
                                              st_2_stereotype_109])
good_news_everyone_42 = Task(task_id=42,
                         subtasks_list=[st_1_good_news_everyone_42,
                                        st_2_good_news_everyone_42])
gwsd_128 = Task(task_id=128, subtasks_list=[st_1_gwsd_128])


# MBIB ###
# st_linguistic = ClassificationSubTask(task_id=11111, id=11111, filename="mbib_linguistic/preprocessed.csv", num_classes=2)
# mbib_lingustic = Task(task_id=11111, subtasks_list=[st_linguistic])

# Create task object
all_tasks = [
babe_10,
cw_hard_03,
me_too_ma_108,
pheme_12,
mdgender_116,
mpqa_103,
stereotype_109,
good_news_everyone_42,
gwsd_128,
]

# Get all subtasks
all_subtasks = list(itertools.chain.from_iterable(t.subtasks_list for t in all_tasks))

# test_task object  list for debugging
test_tasks = [cw_hard_03, me_too_ma_108, good_news_everyone_42]

test_subtasks = list(itertools.chain.from_iterable(t.subtasks_list for t in test_tasks))

# Task families
media_bias = [babe_10]
subjective_bias = [cw_hard_03]
hate_speech = [me_too_ma_108]
gender_bias = [mdgender_116]
sentiment_analysis = [mpqa_103]
fake_news = [pheme_12]
group_bias = [stereotype_109]
emotionality = [good_news_everyone_42]
stance_detection = [gwsd_128]
#mlm = [mlm_0]