"""
Logistic Regression Implementation
"""

import numpy as np
from loguru import logger


class LogisticRegression:
    # ref : https://domybestinlife.tistory.com/m/272

    def __init__(self, input_data, target_output, learn_rate):
        # all initial training data is fed on instance creation here

        # x would be at least 2 dim most time, but y could be 1 dim or more
        # Depending on datasets. So make y least 2 dim by un-squeezing over axis 1.
        # wouldn't need such consideration if I'm only going to use this on titanic dataset.
        self.xs = input_data
        self.ys = target_output

        self.w = np.random.rand(len(self.xs[0]))
        self.lr = learn_rate

    def sigmoid(self, x):
        # aka logistic?
        return 1 / (1 + np.exp(-x))

    def cost(self, h, y):
        # -(1 / m) * [yi * log(hi) + (1 - yi) * log(1 - hi) for i in range(m)]
        #             A---------   B-------------------

        a = y * np.log(h)
        b = (1 - y) * np.log(1 - h)

        # return -(1 / len(h)) * np.sum(a + b)
        return - (a + b)

    def predict(self, x) -> np.ndarray:
        return self.sigmoid(np.dot(self.w.T, x))

    def gradient_decent(self):

        grad = np.zeros_like(self.w)

        for w_idx in range(len(self.w)):
            grad[w_idx] = np.sum([(self.predict(x) - y) * x[w_idx] for x, y in zip(self.xs, self.ys)])

        self.w -= self.lr * np.array(grad)

    def train(self):
        self.gradient_decent()

        return np.mean([self.cost(self.predict(x), y) for x, y in zip(self.xs, self.ys)])


def test(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, epochs: int, lr: float):
    """Logistic Regression 네트워크로 학습 / 테스트 후 정답률 반환.
    멀티프로세싱에 쓰기 좋게 별도의 함수로 분리.

    Args:
        train_x:
        train_y:
        test_x:
        test_y:
        epochs: 학습 반복 수
        lr: 학습률

    Returns:
        (학습 단계별 코스트, 정확도)
    """

    net = LogisticRegression(train_x, train_y, lr)
    costs = []

    for run in range(1, epochs + 1):
        cost = net.train()

        if not run % 100:
            logger.debug(f"Logistic Regression Training [{run}/{epochs}] Done, avg cost: {cost:.6f}")

        costs.append(cost)

    logger.debug("Logistic Regression Network Test start")

    # 정답 수
    hits = 0

    for x, y in zip(test_x, test_y):
        if (net.predict(x) >= 0.5) == y:
            hits += 1

    # 정확도 계산 및 출력 후 반환
    accuracy = hits / len(test_y)
    logger.debug(f"Logistic Regression Network Test done, acc: {accuracy * 100:.2f}%")

    return costs, accuracy

