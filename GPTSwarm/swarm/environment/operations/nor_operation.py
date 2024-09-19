#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry
import json


class NoRAnswer(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Directly output an answer.",
                 id=None):
        super().__init__(operation_description, id, True)
        print(f"{{.py}} creating Node with {domain=}, {model_name=},")
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        return self.__class__.__name__

    def meta_prompt(self, input, meta_init=False):

        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()
        # Change!!!
        prompt = self.prompt_set.get_rag_answer_prompt(question=task)    

        if meta_init:
            pass #TODO

        return role, constraint, prompt


    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        inputs = []
        print(f"{{rag_answer.py}} execute() with {len(node_inputs)} inputs...")
        print(f"{{rag_answer.py}} {node_inputs=}")
        wanted_question_id = node_inputs[0]["task"]


        print(f"{{rag_answer.py}} wanted_question_id: {wanted_question_id}")


        def load_data(file_path):
            data = []
            with open(file_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
            return data

        def find_answers(question_id, 
                         train_file="/home/mhoveyda/AdaptiveQA/Adaptive-RAG/DATA_FOR_CMAB/Train_Data_For_CMAB_augmented_with_confidence.json",
                         test_file='/home/mhoveyda/AdaptiveQA/Adaptive-RAG/DATA_FOR_CMAB/Test_Data_For_CMAB_augmented_with_confidence.json'
                          ):
            
            # Load data from files
            train_data = load_data(train_file)
            test_data = load_data(test_file)
            
            # Combine data
            combined_data = train_data + test_data
            
            # Search for the question_id in the combined data
            for item in combined_data:
                if item['question_id'] == question_id:
                    # return {
                    #     "NoR_predicted_answer": item.get("NoR_predicted_answer"),
                    #     "OneR_predicted_answer": item.get("OneR_predicted_answer"),
                    #     "IRCoT_predicted_answer": item.get("IRCoT_predicted_answer")
                    # }
                    return item.get("NoR_predicted_answer")
            
            return None
    
        answer = find_answers(wanted_question_id)

        print(f"{{rag_answer.py}} !answer found: {answer}, {node_inputs=}")


        for input in node_inputs:
            # role, constraint, prompt= self.meta_prompt(input, meta_init=False)
            # message = [Message(role="system", content=f"You are a {role}. {constraint}"),
            #         Message(role="user", content=prompt)]
            # response = await self.llm.agen(message) The original code
            # print(f"\n{{rag_answer.py}} calling self.llm.agen with \n\t{message=}, {self.max_token=}")
            # response = await self.llm.agen(message,max_tokens=self.max_token)
            # response = self.llm.gen(message,max_tokens=self.max_token) 
            # print(f"{{rag_answer.py}} {response=}\n")
            response = answer
            prompt = "dummy prompt"
 

            _memory = {
                "operation": self.node_name,
                #"task_id": input["task_id"], 
                "task": input["task"],
                "files": input.get("files", []),
                "input": input["task"],
                "subtask": prompt,
                "output": response,
                "format": "natural language"
            }

            # self.log()
            inputs.append(_memory)
        return inputs

