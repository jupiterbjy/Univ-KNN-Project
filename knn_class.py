"""
KNN Implementation
"""

from typing import Tuple, List
from collections import Counter

import numpy as np


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

    # def majority_vote(self, point) -> int:
    #     """Determines class by given points via majority vote.
    #
    #     Args:
    #         point: new unclassified point(data)
    #
    #     Returns:
    #         Classification result
    #     """
    #
    #     # counts how many target appeared, then get most common class(target).
    #     return Counter((tgt for tgt, _ in self.get_knn(point))).most_common(1)[0][0]

    def majority_vote_weighted(self, point) -> int:
        """Determines class by given points via weighted majority vote.

        Args:
            point: new unclassified point(data)

        Returns:
            Classification result
        """
        counter = {}

        # add up inverse distance for each different classes(targets).
        for target, dist in self.get_knn(point):
            counter[target] = counter.setdefault(target, 0) + (1/dist)

        # sort the dictionary, and fetch last(largest) element.
        return sorted(counter.items(), key=lambda x: x[1])[-1][0]
