"""
Deep Neural Network 구현체
"""

import itertools
from typing import Union

import numpy as np
from loguru import logger

from . import NetworkABC, repeat_training, validate, draw_loss


class DNN(NetworkABC):
    def __init__(self, input_data, target_output, hidden_count, hidden_size, learn_rate, momentum=0.9):
        """[Affine -> ReLU] -> Softmax 신경망 구현 클래스.
        [Affine -> ReLU] 계층의 수를 자유롭게 조절 가능.

        Args:
            input_data: 학습 데이터 입력 값들
            target_output: 학습 데이터 라벨 값들
            hidden_count: 히든 레이어 수
            hidden_size: 히든 레이어 차원
            learn_rate: 학습률
            momentum: Momentum 최적화 알고리즘에 쓰일 관성 배율
        """

        self.xs = input_data
        self.ys = target_output

        # 입력, 출력 크기 저장
        self.input_size = len(self.xs[0])
        self.output_size = 1

        # 히든 레이어 크기를 담는 배열. 가독성을 위해 이름 붙임.
        self.hidden_sizes = list(itertools.repeat(hidden_size, hidden_count))

        # TODO: 다른 가중치 초기화 방법 사용
        # 가중치 배열 초기화. 계층 수가 고정이 아니므로 제너레이터 구문 사용.
        # Python 3.10 부터 추가된 itertools.pairwise 를 사용해 2개씩 묶어서 행렬을 만듦.
        self.ws = [
            np.random.randn(size_1, size_2)
            for size_1, size_2 in itertools.pairwise(
                (self.input_size, *self.hidden_sizes, self.output_size)
            )
        ]

        # 편향값 초기화.
        self.bs = [
            np.zeros(size) for size in (*self.hidden_sizes, self.output_size)
        ]

        # Affine 레이어 생성
        self.weighted_layers = [Affine(w, b) for w, b in zip(self.ws, self.bs)]

        # Affine 사이에 ReLU를 끼워 넣은 리스트 생성
        self.layers = []
        for layer in self.weighted_layers:
            self.layers.extend((layer, ReLU()))

        # 리스트 맨 마지막에 남는 ReLU 제거
        self.layers.pop()

        # Softmax 를 최하위 계층으로 사용
        self.layer_last = Softmax()

        # 죄적화에 Momentum 사용
        self.optimizer = Momentum(self.ws, learn_rate, momentum)

    @property
    def structure(self):
        """모델 구조를 문자열로 반환하는 프로퍼티"""

        # 각 레이어 차원을 배열로 모아 화살표로 연결. 입력 데이터가 축소되는 과정을 담기 위함.
        size_structure = [self.input_size, *self.hidden_sizes, self.output_size]
        nameplate = f"DNN ({' -> '.join(map(str, size_structure))})\n"

        # 레이어 이름 나열
        structures = " > ".join(layer.name for layer in self.layers)
        return nameplate + structures + f" > {self.layer_last.name}"

    @staticmethod
    def cost(h, y) -> float:
        """교차 엔트로피 오차 (Cross Entropy Error)를 구하는 함수.

        Args:
            h: 모델의 예측값
            y: 라벨

        Returns:
            비용
        """

        # log(0) 을 막기 위해 작은 값(델타) 더하기
        return -np.sum(y * np.log(h + 1e-7))

    def predict(self, x) -> float:
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def optimize(self, x, y) -> float:
        """최적화 함수. 여기선 Momentum 을 사용.

        Args:
            x: 학습 데이터 입력
            y: 학습 데이터 라벨

        Returns:
            교차 엔트로피 오차
        """

        # 순전파
        h = self.predict(x)
        cost = self.cost(self.layer_last.forward(h), y)

        # 역전파
        dout = self.layer_last.backward(1, y)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # gradient 저장
        grads_w = [layer.dw for layer in self.weighted_layers]
        grads_b = [layer.db for layer in self.weighted_layers]

        # gradient 반영
        self.optimizer.update(self.ws, grads_w, self.bs, grads_b)

        # 오차 반환
        return cost

    def train(self):
        """경사 하강법을 이용해 학습을 진행하는 함수

        Returns:
            전체 학습 데이터의 비용 평균 반환
        """

        return np.mean([self.optimize(x, y) for x, y in zip(self.xs, self.ys)])


class Module:
    """신경망을 구성하는 모듈의 추상 베이스 클래스"""

    @property
    def name(self):
        return type(self).__name__

    # softmax 계층에서 다르게 쓰는 인자가 있어 *args 사용
    def forward(self, *args) -> np.ndarray:
        raise NotImplementedError

    def backward(self, *args) -> np.ndarray:
        raise NotImplementedError


class ReLU(Module):
    # 교재 '밑바닥부터 시작하는 딥러닝' 기반

    def __init__(self):
        self.mask: Union[np.ndarray, None] = None

    def forward(self, x) -> np.ndarray:
        self.mask = (x <= 0)

        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout) -> np.ndarray:
        dout[self.mask] = 0
        # ^ dx
        return dout


class Sigmoid(Module):
    def __init__(self):
        self.out: Union[np.ndarray, None] = None

    def forward(self, x) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine(Module):
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.w = weight
        self.b = bias
        self.x: Union[np.ndarray, None] = None

        # 미분값
        self.dw: Union[np.ndarray, None] = None
        self.db: Union[np.ndarray, None] = None

    def forward(self, x) -> np.ndarray:
        self.x = x
        out = np.dot(x, self.w) + self.b

        return out

    def backward(self, dout) -> np.ndarray:
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Softmax(Module):
    """Softmax 구현 모듈. 인터페이스를 다른 네트워크와 통일하기 위해
    비용 계산 부분을 제거."""
    def __init__(self):
        self.y: Union[np.ndarray, None] = None

    @staticmethod
    def softmax(x):
        # e^x / sum(e^x)
        # 이때 오버플로를 막기 위해 x 중 가장 큰 값을 찾아서 뺌.
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def forward(self, x) -> np.ndarray:
        self.y = self.softmax(x)

        return self.y

    def backward(self, dout, t) -> np.ndarray:
        dx = (self.y - t)

        return dx


class Momentum:
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.moment = momentum
        self.v = [np.zeros_like(param) for param in params]

    def update(self, weights, grads_w, biases, grads_b):
        """가중치 & 편향 갱신 함수

        Args:
            weights: 각 레이어 가중치 배열을 담은 리스트
            grads_w: 각 레이어 가중치 기울기를 담은 리스트
            biases: 각 레이어 편향 배열을 담은 리스트
            grads_b: 각 레이어 편향 기울기를 담은 리스트

        Returns:

        """

        # 가중치 & 편향 갱신
        for params, grads in zip((weights, biases), (grads_w, grads_b)):
            for idx, (vel, grad) in enumerate(zip(self.v, grads)):
                self.v[idx] = self.moment * vel - self.lr * grad
                params[idx] += self.v[idx]


def test(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    hidden_count: int,
    hidden_size: int,
    epochs: int,
    lr: float,
    moment: float = 0.9,
):
    """Linear Regression 네트워크로 학습 / 테스트 & 손실 그래프 저장.
    멀티프로세싱에 쓰기 좋게 별도의 함수로 분리.

    Args:
        train_x: 학습 데이터 입력 값들
        train_y: 학습 데이터 라벨 값들
        test_x: 테스트 데이터 입력 값들
        test_y: 테스트 데이터 라벨 값들
        hidden_count: 히든 레이어 수
        hidden_size: 히든 레이어 크기
        epochs: 학습 반복 수
        lr: 학습률
        moment: Momentum optimizer 에서 사용할 관성 배율

    Returns:
        (네트워크 이름, 정확도)
    """

    net = DNN(train_x, train_y, hidden_count, hidden_size, lr, moment)

    logger.info(f"DNN initialized with following structure:\n{net.structure}")

    costs = repeat_training(net, epochs)
    accuracy = validate(net, test_x, test_y)

    draw_loss(net, costs)

    return net.name, accuracy
