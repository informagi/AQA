from swarm.graph import Graph
from swarm.environment.operations.oner_operation import OneRAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('OneRAgent')
class OneRAgent(Graph):
    def build_graph(self):

        oner_answer = OneRAnswer(self.domain, self.model_name)

        self.input_nodes = [oner_answer]
        self.output_nodes = [oner_answer]

        self.add_node(oner_answer)
