"""
KNN Implementation
"""

from typing import Tuple, List, Sequence

import numpy as np
from loguru import logger


class KNN:
    def __init__(self, k: int, features, target):
        self.k = k
        self.x = features
        self.y = target

    @staticmethod
    def dist_squared(p1, p2) -> float:
        """Calculates sum of squared distance features between given points.

        Args:
            p1: point 1
            p2: point 2

        Returns:
            Sum of each feature's squared distances.
        """

        p1_n = np.array(p1)
        p2_n = np.array(p2)

        # zip given points then calculate sum of each feature's squared distance
        return sum(np.square(p1_n - p2_n))

    def get_knn(self, p) -> List[Tuple[int, float]]:
        """Finds class(target) & distance of knn points from given point.

        Args:
            p: point

        Returns:
            List of (class, distance).
        """

        # pair (target, distance) for each trained point against given point.
        target_dist_pairs = [(target, self.dist_squared(p, trained_p)) for trained_p, target in zip(self.x, self.y)]

        # sort via distance, and slice first k records
        return sorted(target_dist_pairs, key=lambda x: x[-1])[:self.k]

    def predict(self, point) -> int:
        """Determines class by given points via weighted majority vote.

        Args:
            point: new unclassified point(data)

        Returns:
            Classification result
        """
        counter = {}

        # add up inverse distance for each different classes(targets).
        for target, dist in self.get_knn(point):

            # dist could be 1, add dummy value
            counter[target] = counter.setdefault(target, 0) + (1 / (dist + 1e-6))

        # sort the dictionary, and fetch last(largest) element.
        return sorted(counter.items(), key=lambda x: x[1])[-1][0]


def test(train_x: Sequence, train_y: Sequence, test_x: Sequence, test_y: Sequence, k: int):
    """주어진 K 값에 따른 KNN 네트워크로 테스트 후 정답률 반환.
    멀티프로세싱에 쓰기 좋게 별도의 함수로 분리.

    Args:
        train_x:
        train_y:
        test_x:
        test_y:
        k:

    Returns:
        (코스트, 정확도) - 여기에 코스트는 없으므로 빈 리스트.
    """

    logger.debug(f"{k}-nn Network Test start")

    net = KNN(k, train_x, train_y)

    # 정답 수
    hits = 0

    # 테스트 수행
    for x, y in zip(test_x, test_y):

        # 가중치를 고려한 다수결을 사용하여 예측 후 카운터 +1
        if net.predict(x) == y:
            hits += 1

    # 정확도 계산 및 출력 후 반환
    accuracy = hits / len(test_y)
    logger.debug(f"{k}-nn Network Test done, acc: {accuracy * 100:.2f}%")

    return [], accuracy
