"""
linear Regression Implementation
"""

import numpy as np

from . import Network, repeat_training, validate, draw_loss


class LinearRegression(Network):
    def __init__(self, input_data, target_output, learn_rate):
        """Linear Regression 구현 클래스

        Args:
            input_data: 학습 데이터 입력 값들
            target_output: 학습 데이터 라벨 값들
            learn_rate: 학습률
        """

        # 모든 학습 데이터는 인스턴스 생성시점에 전달됨.
        self.xs = input_data
        self.ys = target_output

        # 학습 데이터의 입력 값 차원 길이 만큼의 0 배열로 가중치 초기화.
        self.w = np.zeros(len(self.xs[0]))
        self.lr = learn_rate

    @staticmethod
    def cost(h, y):
        """비용 함수. 최적화 된 경사 하강법 함수에 코스트 계산이 포함 되어 있어
        사용하지는 않음. 순전히 오차의 정도를 측정하기 위한 함수.

        Args:
            h: 모델의 예측값
            y: 라벨

        Returns:
            비용
        """

        # (1 / m) * [(hi - yi) ** 2 for i in range(m)]
        # = mean([(hi - yi) ** 2 for i in range(m)])
        # => Mean Square Error

        # => mean((wx-y)**2)
        return np.mean(np.square(y - h))

    def predict(self, x) -> float:
        """입력 값을 받아 모델의 예측값 반환.

        Args:
            x: 입력값

        Returns:
            예측값
        """

        # np.dot 타입 힌트가 잘못됨. ndarray 와 float 둘다 반환할 수 있어 강제로 타입 명시.
        # noinspection PyTypeChecker
        return np.dot(self.w.T, x)

    def gradient_decent(self):
        """경사 하강법 함수. Logistic Regression 과 구조적으로 동일"""

        grad = np.zeros_like(self.w)

        # ∂cost(Θ) / ∂Θ = (1 / m) * sum([(hi - yi) * xi for i in range(m)])
        #               = mean([(hi - yi) * xi for i in range(m)])
        for w_idx in range(len(self.w)):
            grad[w_idx] = np.mean(
                [(self.predict(x) - y) * x[w_idx] for x, y in zip(self.xs, self.ys)]
            )

        self.w -= self.lr * np.array(grad)

    def train(self):
        """경사 하강법을 이용해 학습을 진행하는 함수

        Returns:
            전체 학습 데이터의 비용 평균
        """

        self.gradient_decent()

        return np.mean(
            [self.cost(self.predict(x), y) for x, y in zip(self.xs, self.ys)]
        )


def test(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int,
    lr: float,
):
    """Linear Regression 네트워크로 학습 / 테스트 & 손실 그래프 저장.
    멀티프로세싱에 쓰기 좋게 별도의 함수로 분리.

    Args:
        train_x: 학습 데이터 입력 값들
        train_y: 학습 데이터 라벨 값들
        test_x: 테스트 데이터 입력 값들
        test_y: 테스트 데이터 라벨 값들
        epochs: 학습 반복 수
        lr: 학습률

    Returns:
        (네트워크 이름, 정확도)
    """

    net = LinearRegression(train_x, train_y, lr)

    costs = repeat_training(net, epochs)
    accuracy = validate(net, test_x, test_y)

    draw_loss(net, costs)

    return net.name, accuracy
