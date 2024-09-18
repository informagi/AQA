from swarm.graph import Graph
from swarm.environment.operations.ircot_operation import IRCoTAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('IRCoTAgent')
class IRCoTAgent(Graph):
    def build_graph(self):


        ircot_answer = IRCoTAnswer(self.domain, self.model_name)

        self.input_nodes = [ircot_answer]
        self.output_nodes = [ircot_answer]

        self.add_node(ircot_answer)
