"""
Deep Neural Network 구현체
"""

import itertools
from typing import Union

import numpy as np
from loguru import logger

from . import NetworkABC, repeat_training, validate, draw_loss


class DNN(NetworkABC):
    def __init__(self, input_data, target_output, hidden_count, hidden_size, learn_rate, momentum=0.9, batch_size=10):
        """[Affine -> ReLU] -> Softmax 신경망 구현 클래스.
        [Affine -> ReLU] 계층의 수를 자유롭게 조절 가능.

        Args:
            input_data: 학습 데이터 입력 값들
            target_output: 학습 데이터 라벨 값들
            hidden_count: 히든 레이어 수
            hidden_size: 히든 레이어 차원
            learn_rate: 학습률
            momentum: Momentum 최적화 알고리즘에 쓰일 관성 배율
            batch_size: 미니 배치 크기
        """

        self.xs: np.ndarray = input_data
        # self.ys = np.expand_dims(target_output, axis=1)
        self.ys = np.array([[0, 1] if y == 1 else [1, 0] for y in target_output])
        self.batch_size = batch_size

        # 입력, 출력 크기 저장
        self.input_size = len(self.xs[0])
        self.output_size = 2

        # 히든 레이어 크기를 담는 배열. 가독성을 위해 이름 붙임.
        # 실제 생기는 Affine 층 수는 +1 개 이므로 -1을 미리 함.
        self.hidden_sizes = list(itertools.repeat(hidden_size, hidden_count - 1))

        # 가중치 배열 초기화. 계층 수가 고정이 아니므로 제너레이터 구문 사용. 이때 He 초기화 사용.
        # Python 3.10 부터 추가된 itertools.pairwise 를 사용해 2개씩 묶어서 행렬을 만듦.
        self.ws = [
            np.random.randn(size_1, size_2) / np.sqrt(2 / size_1)
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

        # Affine 사이에 Dropout, ReLU를 끼워 넣은 리스트 생성
        self.layers = []
        for layer in self.weighted_layers:
            # self.layers.extend((layer, Sigmoid()))
            self.layers.extend((layer, ReLU(), Dropout()))

        # 리스트 맨 마지막에 남는 Dropout 제거
        self.layers = self.layers[:-2]

        # Softmax 를 최하위 계층으로 사용
        self.layer_last = SoftmaxWithLoss()

        # 죄적화에 Momentum 사용
        self.optimizer = SGD(learn_rate)
        # self.optimizer = Momentum(self.ws, self.bs, learn_rate, momentum)

        # 학습 중인지 나타 내는 플래그. Dropout 에 사용.
        self.training_mode = True

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
        """오차 제곱합 비용 함수.

        Args:
            h: 모델의 예측값
            y: 라벨

        Returns:
            비용
        """

        return 0.5 * np.sum((h - y) ** 2)

    def predict(self, x):
        """입력 값을 받아 모델의 예측값 반환.

        Args:
            x: 입력값

        Returns:
            예측값
        """

        for layer in self.layers:
            x = layer.forward(x, self.training_mode)

        return np.argmax(x)

    def forward(self, x, t):
        """순전파 과정.
        위의 predict 함수와 차이는 softmax 계층을 쓰는지, np.squeeze 를 하는지 여부.
        본래 predict 함수로 최대한 해보려 했으나, 이를 위해 다른 부분들이 복잡해져서 분리.

        Args:
            x: 입력값
            t: 라벨값

        Returns:
            오차
        """

        for layer in self.layers:
            x = layer.forward(x, self.training_mode)

        return self.layer_last.forward(x, t)

    def optimize(self, x, y) -> float:
        """최적화 함수. 여기선 Momentum 을 사용.

        Args:
            x: 학습 데이터 입력
            y: 학습 데이터 라벨

        Returns:
            교차 엔트로피 오차
        """

        # 순전파
        h = self.forward(x, y)
        cost = self.cost(h, y) / self.batch_size

        # 역전파
        dout = self.layer_last.backward(1)
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

        data_length = self.xs.shape[0]

        costs = []

        for _ in range(data_length // self.batch_size):
            batch_mask = np.random.choice(data_length, self.batch_size)
            x_batch = self.xs[batch_mask]
            y_batch = self.ys[batch_mask]

            costs.append(self.optimize(x_batch, y_batch))

        return np.mean(costs)


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


# -------------------------------
# 이하 모듈 정의
# 교재 '밑바닥부터 시작하는 딥러닝' 기반

class ReLU(Module):
    """Leaky Rectified Linear Unit 구현."""

    def __init__(self):
        self.mask: Union[np.ndarray, None] = None

    def forward(self, x, *_) -> np.ndarray:
        self.mask = (x <= 0)

        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout) -> np.ndarray:
        dout[self.mask] = 0

        return dout


class Sigmoid(Module):
    """Sigmoid 구현 - 사용하지 않음"""

    def __init__(self):
        self.out: Union[np.ndarray, None] = None

    def forward(self, x, *_) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine(Module):
    """Affine 구현 (Dense?)"""

    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.w = weight
        self.b = bias
        self.x: Union[np.ndarray, None] = None

        # 미분값
        self.dw: Union[np.ndarray, None] = None
        self.db: Union[np.ndarray, None] = None

    def forward(self, x, *_) -> np.ndarray:
        self.x = x
        out = np.dot(x, self.w) + self.b

        return out

    def backward(self, dout) -> np.ndarray:
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss(Module):
    """Softmax 에 교차 크로스 엔트로피 구현"""
    def __init__(self):
        self.y: Union[np.ndarray, None] = None
        self.t: Union[np.ndarray, None] = None

    @staticmethod
    def cross_entropy_err(h, y):
        return -np.sum(y * np.log(h + 1e-7))

    @staticmethod
    def softmax(x):
        # e^x / sum(e^x)
        # 이때 오버플로를 막기 위해 x 중 가장 큰 값을 찾아서 뺌.
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def forward(self, x, t) -> np.ndarray:
        self.t = t
        self.y = self.softmax(x)

        return self.cross_entropy_err(self.y, t)

    def backward(self, dout=1) -> np.ndarray:
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class Dropout(Module):
    """Dropout 구현"""

    def __init__(self, dropout_rate=0.5):
        self.drop_r = dropout_rate
        self.mask: Union[np.ndarray, None] = None

    def forward(self, x, training) -> np.ndarray:
        if training:
            self.mask = np.random.rand(*x.shape) > self.drop_r
            return x * self.mask

        return x * (1.0 - self.drop_r)

    def backward(self, dout) -> np.ndarray:
        return dout * self.mask


class Momentum:
    """Momentum optimizer 구현"""

    def __init__(self, weights, biases, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.moment = momentum

        # 가중치, 편향 각각의 속도를 분리해 저장
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

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
        for velocities, params, grads in zip((self.v_w, self.v_b), (weights, biases), (grads_w, grads_b)):
            # 가중치와 편향을 따로 계산하므로 for 루프로 한번 더 감쌈

            for idx, (velocity, grad) in enumerate(zip(velocities, grads)):
                velocities[idx] = self.moment * velocity - self.lr * grad
                params[idx] += velocities[idx]


class SGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, weights, grads_w, biases, grads_b):

        # 가중치 & 편향 갱신
        for idx, grad in enumerate(grads_w):
            weights[idx] -= grad * self.lr

        for idx, grad in enumerate(grads_b):
            biases[idx] -= grad * self.lr


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
    batch_size: int = 10,
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
        batch_size: 미니 배치 크기

    Returns:
        (네트워크 이름, 정확도)
    """

    net = DNN(train_x, train_y, hidden_count, hidden_size, lr, moment, batch_size)

    logger.info(f"DNN initialized with following structure:\n{net.structure}")

    net.training_mode = True
    costs = repeat_training(net, epochs, epochs // 10)

    net.training_mode = False
    accuracy = validate(net, test_x, test_y)

    draw_loss(net, costs)

    return net.name, accuracy
