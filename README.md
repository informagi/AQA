# AQA: Adaptive Question Answering in a Society of LLMs

This is the repository for ***AQA***, a novel adaptive question answering framework that uniquely frames the adaptive QA problem as a contextual multi-armed bandit problem, where the action space is a set of graph structures among LLM agents that describe their interactions. Thereby, the AQA framework dynamically orchestrates the collaboration of multiple agents in response to specific question characteristics.

AQA is built based on various codebases/papers;

1. [GPTSwarm](https://github.com/metauto-ai/GPTSwarm/tree/main): As a framework for formulating the organization of different modules/models as a composite-graph
2. [IRCoT](https://github.com/StonyBrookNLP/ircot): As a framework for defining agents in the composite-graph (i.e. vanilla, OneR, IRCoT agents)
3. [Adaptive_RAG](https://github.com/starsuzi/Adaptive-RAG): We use the dataset introduced in this codebase as it comes with questions with complexity level attributes

In other words, agents are defined/implemented using [IRCoT](https://github.com/StonyBrookNLP/ircot), the swarms (composit-graphs) are designed using [GPTSwarm](https://github.com/metauto-ai/GPTSwarm/tree/main). Formulating Adaptive QA as a Contextual Multi-Armed Bandit problem, we then use the data from [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) to train and evaluate our [LinUCB](https://arxiv.org/pdf/1003.0146) model.