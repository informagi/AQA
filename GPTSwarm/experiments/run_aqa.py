

import asyncio
from typing import Union, Literal, Optional
import argparse

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator import Evaluator
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.evaluator.datasets.aqa_dataset import AQADataset

# set random seed as 42
# import random
# random.seed(42)
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False
# import numpy as np
# np.random.seed(42)

import random
import numpy as np
import torch
import logging
# import tensorflow as tf

# Setting the seed for various random number generators
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# tf.random.set_seed(42)

log_file_path = "../LOGS/GPTSwarm/logs.txt"


logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and N adversarial.")

    parser.add_argument('--num-iterations', type=int, default=200,
                        help="Number of optimization iterations. Default 200.")

    parser.add_argument('--model_name', type=str, default=None,
                        help="Model name, None runs the default ChatGPT4.")

    parser.add_argument('--domain', type=str, default="aqa",
                        help="Domain (the same as dataset name), default 'MMLU'")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="Set for a quick debug cycle")

    args = parser.parse_args()
    return args


async def main():

    args = parse_args()
    logging.info(f"\n{{run_witqa.py}} \n\targs: {args}")

    debug: bool = args.debug

    model_name: Optional[str] = args.model_name

    mode: Union[Literal['DirectAnswer'],
                Literal['FullConnectedSwarm'],
                Literal['RandomSwarm'],
                Literal['OptimizedSwarm']]

    mode = args.mode

    strategy = MergingStrategy.MajorityVote
    logging.info(f"\n\t{{run_aqa.py}} \n\tstrategy: {strategy}")

    domain: str = args.domain
    

    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        N = args.num_truthful_agents

        M = N

        # agent_name_list = N * ["IO"] + M * ["AdversarialAgent"]
        # agent_name_list = N * ["IO"] + M * ["IO"] # Mohanna
        # agent_name_list = N * ["IO"] + M * ["RAGAgent"]
        agent_name_list = N * ["NoRAgent"] + M * ["OneRAgent"] + M * ["IRCoTAgent"]


        logging.info(f"\n\t{{run_witqa.py}} \n\tagent_name_list: {agent_name_list}")


        # swarm_name = f"{N}true_{M}adv"
        # swarm_name = f"{N}true_{M}ture" # Mohanna
        swarm_name = f"{N}NoR_{M}OneR_{M}IRCoT"

        logging.info(f"\n\t{{run_witqa.py}} initializing swarm...")
        swarm = Swarm(
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}"
    logging.info(f"\n\t{{run_aqa.py}} tag: {tag}")


    logging.info(f"\n\t{{run_aqa.py}} Loading train (dev)")
    dataset_train = AQADataset('dev')
    logging.info(f"\n\t{{run_witqa.py}} Loading val")
    dataset_val = AQADataset('val')


    evaluator = Evaluator(
        swarm,
        dataset_train,
        dataset_val,
        model_name=model_name,
        enable_tensorboard = mode=='OptimizedSwarm',
        enable_artifacts=True,
        tensorboard_tag=tag)
    # debug = True
    limit_questions = 2 if debug else 210
    # # # limit_questions = 1 if debug else 153

    logging.info(f"\n\t{{run_witqa.py}} limit_questions: {limit_questions}")

    if mode == 'DirectAnswer':
        logging.info(f"\n\t{{run_witqa.py}} evaluating direct answer...")
        score = await evaluator.evaluate_direct_answer(
            limit_questions=limit_questions)
        
    elif mode == 'FullConnectedSwarm':
        logging.info(f"\n\t{{run_witqa.py}} evaluating full connected swarm...")
        score = await evaluator.evaluate_swarm(
            mode='full_connected_swarm',
            limit_questions=limit_questions)
        

    elif mode == 'RandomSwarm':
        logging.info(f"\n\t{{run_witqa.py}} evaluating random swarm...")
        score = await evaluator.evaluate_swarm(
            mode='randomly_connected_swarm',
            limit_questions=limit_questions)
        
    elif mode == 'OptimizedSwarm':
        logging.info(f"\n\t{{run_witqa.py}} optimizing swarm...")

        # num_iters = 5 if debug else args.num_iterations
        num_iters = 1 if debug else args.num_iterations

        logging.info(f"\n\t{{run_witqa.py}} num_iters: {num_iters}")

        lr = 0.1

        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr)
        logging.info(f"\n\t{{run_witqa.py}} edge_probs: \n{edge_probs}")
        logging.info(f"\n\t{{run_witqa.py}} evaluating swarm with optimized edge_probs...")
        score = await evaluator.evaluate_swarm(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            )

    else:
        raise Exception(f"Unsupported mode {mode}")

    logging.info(f"Score: {score}")


if __name__ == "__main__":
    logging.info(f"{{run_aqa.py}}")
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Exception: {e}", exc_info=True)
        raise e
