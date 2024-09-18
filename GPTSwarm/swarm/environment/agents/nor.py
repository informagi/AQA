from swarm.graph import Graph
from swarm.environment.operations.nor_operation import NoRAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('NoRAgent')
class NoRAgent(Graph):
    def build_graph(self):


        nor_answer = NoRAnswer(self.domain, self.model_name)

        self.input_nodes = [nor_answer]
        self.output_nodes = [nor_answer]

        self.add_node(nor_answer)
