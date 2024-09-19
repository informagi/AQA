import glob
import pandas as pd
from typing import Union, List, Literal
import numpy as np
import json
import re
from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput


# set random seed as 42
import random
random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(42)
class AQADataset(BaseDataset):
    def __init__(self,
        split: Union[Literal['dev'], # train
                      Literal['val'], 
                    #  Literal['test']
                     ]
                     ,
        ) -> None:

        self._split = split

        data_path = f"datasets/AQA/{self._split}.tsv"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'aqa'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        # print("Number of topics: ", len(csv_paths))

        # names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']
        # names = ['question', 'correct_answer', 'IR_Oracle']
        names = ['question_id', 'question_text', 'gold_answers', 'complexity_label', 'NoR_predicted_answer', 'NoR_time_taken', 'OneR_predicted_answer', 'OneR_time_taken', 'IRCoT_predicted_answer', 'IRCoT_steps_taken', 'IRCoT_time_taken', 'NoR_evaluation_results', 'OneR_evaluation_results', 'IRCoT_evaluation_results', 'NoR_number_of_subkeys', 'NoR_total_run_time_in_seconds', 'NoR_average_confidence_score_among_all_subkeys', 'NoR_average_confidence_score_of_the_last_subkey', 'OneR_number_of_subkeys', 'OneR_total_run_time_in_seconds', 'OneR_average_confidence_score_among_all_subkeys', 'OneR_average_confidence_score_of_the_last_subkey', 'IRCoT_number_of_subkeys', 'IRCoT_total_run_time_in_seconds', 'IRCoT_average_confidence_score_among_all_subkeys', 'IRCoT_average_confidence_score_of_the_last_subkey']


        total_df = pd.DataFrame(columns=names)
        # single_df = pd.read_csv(data_path, header=None, names=names)
        single_df = pd.read_csv(data_path, header=None, names=names, delimiter='\t', skiprows=1)

        total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))
        print(f"First question: \n{total_df.iloc[0]}\n")

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        print(f"In record_to_swarm_input in aqa_dataset.py")
        # demo_question = (
        #     f"Context: {record['IR_Oracle']}\n"
        #     f"Question: {record['question']}\n"
        #     )
        demo_question = record['question_id']
        # demo_context = f"{record['IR_Oracle']}\n"
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        # if isinstance(answer, list):
        #     if len(answer) > 0:
        #         answer = answer[0]
        #     else:
        #         answer = ""
        # if not isinstance(answer, str):
        #     raise Exception("Expected string")
        # if len(answer) > 0:
        #     try:
        #         parsed_answer = json.loads(answer)
        #         answer = parsed_answer["A"]  
        #         if isinstance(answer, list):
        #             if len(answer) > 0:
        #                 answer = answer[0]  
        #             else:
        #                 answer = ""  # Set answer to empty string if the list is empty

        #     except json.JSONDecodeError:
        #         # Regex to match the string format: {"A": "Some Name"} or "A": "Some Name"
        #         pattern = r'\{"?A"?:\s*"([^"]+)"\}'
        #         match = re.search(pattern, answer)
        #         if match:
        #             answer = match.group(1)
        #         else:
        #             answer = "N"  # Fallback to "N" if parsing fails and no match is found.
        #     except KeyError:
        #         answer = "N"
        
        # # If answer had \' in it replace it with nothing
        # answer = answer.replace("\'", "")

        return answer


    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        # correct_answer = record['correct_answer']
        correct_answer = record['gold_answers']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
