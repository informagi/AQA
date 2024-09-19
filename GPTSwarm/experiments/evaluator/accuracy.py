
# class Accuracy:
#     def __init__(self):
#         self._num_correct = 0
#         self._num_total = 0
    
#     def update(self, predicted: str, target: str) -> None:
#         is_correct = predicted == target
#         self._num_correct += int(is_correct)
#         self._num_total += 1

#     def get(self) -> float:
#         return self._num_correct / self._num_total

#     def print(self):
#         accuracy = self.get()
#         print(f"Accuracy: {accuracy*100:.1f}% "
#               f"({self._num_correct}/{self._num_total})")

import ast

from evaluator.squad_answer_em_f1 import SquadAnswerEmF1Metric, normalize_answer

class Accuracy:
    def __init__(self):
        self._f1_metric = SquadAnswerEmF1Metric()
    
    def update(self, predicted: str, targets: list) -> None:
        # predicted = ast.literal_eval(predicted)

        # try:
        targets = ast.literal_eval(targets)
        # except Exception as e:
        #     print(f"targets: {targets} Error: {e}")
        #     targets = [targets]
        print(f"predicted: {predicted}, targets: {targets}")
        if isinstance(predicted, list): predicted = predicted[0]
        print(f"predicted: {predicted}, targets: {targets}")
        print(targets)
        norm_predicted = normalize_answer(predicted)
        print(f"types: {type(norm_predicted)}, {type(targets)}")
        norm_targets = [normalize_answer(target) for target in targets]
        print(f"norm_predicted: {norm_predicted}, norm_targets: {norm_targets}")
        self._f1_metric(norm_predicted, norm_targets)

    def get(self) -> float:
        metrics = self._f1_metric.get_metric()
        return metrics['f1']

    def print(self):
        f1_score = self.get()
        print(f"F1 Score: {f1_score*100:.1f}%")

# Example usage
if __name__ == "__main__":
    f1_score = Accuracy()
    f1_score.update("predicted_answer_1", "target_answer_1")
    f1_score.update("predicted_answer_2", "target_answer_2")
    f1_score.print()