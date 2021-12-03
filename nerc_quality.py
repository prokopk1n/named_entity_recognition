from typing import List, Tuple, Dict, Set, Iterable
from collections import defaultdict


class QualityNERC:
    def _append_test_no(self, answers: List[Set[Tuple[int, int, str]]]) -> Set[Tuple[int, int, int, str]]:
        return {(i, *entity) for i, answer in enumerate(answers) for entity in answer}

    def _group_by_type(self, answers: Iterable[Tuple[int, int, int, str]]) -> Dict[str, Set[Tuple[int, int, int, str]]]:
        ret = defaultdict(set)
        for answer in answers:
            ret[answer[-1]].add(answer)
        return ret   # словарь - метка: {(номер текста, начало, конец), ...}

    def _measure(self, grouped_predicted, grouped_expected, ann_type: str, debug=True) -> float:
        exp = grouped_expected[ann_type]
        pred = grouped_predicted.get(ann_type, set())
        correct = exp.intersection(pred)
        p = len(correct) / len(pred) if len(pred) else 0.0
        r = len(correct) / len(exp)
        f = 2 * len(correct) / (len(exp) + len(pred))

        if debug:
            print(f"{ann_type}: P={p}; R={r}; F1={f}")
        return f

    def evaluate(self, predicted: List[Set[Tuple[int, int, str]]], expected: List[Set[Tuple[int, int, str]]], debug = True):
        grouped_expected = self._group_by_type(self._append_test_no(expected))
        grouped_predicted = self._group_by_type(self._append_test_no(predicted))

        f = 0
        for ann_type in grouped_expected:
            res = self._measure(grouped_predicted, grouped_expected, ann_type, debug)
            f += res

        return f / len(grouped_expected)
