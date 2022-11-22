"""
Logistic Regression Implementation
"""

import numpy as np
from loguru import logger


class LogisticRegression:
    # ref : https://domybestinlife.tistory.com/m/272
    log_upper_lim = 100
    log_lower_lim = -100

    def __init__(self, input_data, target_output, learn_rate):
        # all initial training data is fed on instance creation here

        # x would be at least 2 dim most time, but y could be 1 dim or more
        # Depending on datasets. So make y least 2 dim by un-squeezing over axis 1.
        # wouldn't need such consideration if I'm only going to use this on titanic dataset.
        self.x = input_data
        self.y = target_output if target_output.ndim > 1 else np.expand_dims(target_output, 1)

        self.w = np.random.rand(len(self.x[0]), len(self.y[0]))
        self.lr = learn_rate

    def limit_val(self, x):
        """Limits value so that it doesn't overflow or go Inf, Nan."""
        return np.clip(x, self.log_lower_lim, self.log_upper_lim)

    def sigmoid(self, x):
        # aka logistic?
        return 1 / (1 + np.exp(-x))

    def cost(self, h, y):
        # -(1 / m) * [yi * log(hi) + (1 - yi) * log(1 - hi) for i in range(m)]
        #             A---------   B-------------------
        delta = 1e-9

        # since y gets 1 and 0, log goes inf or nan, need to clip & add small value
        a = y * self.limit_val(np.log(h + delta))
        b = (1 - y) * self.limit_val(np.log(1 - h + delta))

        # return -(1 / len(h)) * np.sum(a + b)
        return - (a + b)

    def predict(self, x) -> np.ndarray:
        return self.sigmoid(np.dot(self.w.T, x))

    def gradient_decent(self, x, y):

        grad = np.zeros_like(self.w)
        dim1, dim2 = self.w.shape

        for i1 in range(dim1):
            for i2 in range(dim2):
                grad[i1][i2] = np.mean((self.predict(x[i1]) - y[i1]) * x[i1])

        self.w -= self.lr * np.array(grad)

    def train(self):
        self.gradient_decent(self.x, self.y)

        return np.mean([self.cost(self.predict(x), y) for x, y in zip(self.x, self.y)])


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
        cost_avg = np.mean(net.train())

        if not run % 20:
            logger.debug(f"Logistic Regression Training [{run}/{epochs}] Done, avg cost: {cost_avg}")

        costs.append(cost_avg)

    logger.debug("Logistic Regression Network Test start")

    # 정답 수
    hits = 0

    for x, y in zip(test_x, test_y):
        # this only have 1 dim,
        if (net.predict(x)[0] >= 0.5) == y:
            hits += 1

    # 정확도 계산 및 출력 후 반환
    accuracy = hits / len(test_y)
    logger.debug(f"Logistic Regression Network Test done, acc: {accuracy * 100:.2f}%")

    return costs, accuracy

