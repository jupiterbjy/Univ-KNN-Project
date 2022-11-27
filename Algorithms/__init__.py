"""
각 알고리즘의 테스트 보조 함수들

알고리즘 각각은 한눈에 흐름을 따라갈 수 있게 따로 중복 코드를 빼지 않음.
"""

from typing import List, Tuple
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


GRAPH = pathlib.Path(__file__).parent.parent / "graph"


class Network:
    """기계 학습 네트워크의 베이스 클래스."""

    @property
    def name(self):
        return type(self).__name__

    @staticmethod
    def cost(h, y) -> float:
        pass

    def predict(self, x) -> float:
        pass

    def train(self) -> float:
        pass


def repeat_training(net: Network, epochs: int, log_interval: int = 100) -> List[float]:
    """주기적으로 로깅을 하며 반복 학습

    Args:
        net: 학습에 쓸 모델
        epochs: 반복 학습 횟수
        log_interval: 로그를 출력하는 간격

    Returns:
        각 epoch 당 평균 비용을 담은 배열
    """

    # 주어진 네트워크 모델 인스턴스의 클래스 이름 저장
    net_type = net.name

    # 주어진 학습 횟수의 자릿수 계산. 순전히 보기 예쁘게 하기 위함.
    digit = len(str(log_interval))

    # epoch 당 비용 평균 저장할 리스트
    costs = []

    logger.debug(f"{net_type} Training start")

    # 1 ~ epochs 까지 반복 학습
    for run in range(1, epochs + 1):
        cost = net.train()

        # 100번마다 중간 상황 출력
        if not run % log_interval:
            logger.debug(
                f"{net_type} Training [{run:{digit}}/{epochs}] avg cost: {cost:.6f}"
            )

        costs.append(cost)

    return costs


def validate(net: Network, test_x, test_y) -> float:
    """테스트 데이터를 가지고 정확도 계산.
    모델이 반환한 예측 값을 반올림 하여 라벨과 비교.

    Args:
        net: 테스트 할 모델
        test_x: 테스트 데이터 입력 값들
        test_y: 테스트 데이터 라벨 값들

    Returns:
        정확도 (0 ~ 1)
    """

    # 주어진 네트워크 모델 인스턴스의 클래스 이름 저장
    net_type = net.name

    # 맞춘 개수
    hits = 0

    logger.debug(f"{net_type} Test start")

    # 테스트
    for x, y in zip(test_x, test_y):
        if round(net.predict(x)) == y:
            hits += 1

    accuracy = hits / len(test_y)
    logger.debug(f"{net_type} Test done, acc: {accuracy * 100:.2f}%")

    return accuracy


def draw_loss(net: Network, cost: List[float]):
    """코스트 값 추이 그래프를 그려 파일로 저장.
    여러 함수를 묶어서 그릴 생각을 했지만, KNN 때문에 개별 그리기로 결정.

    Args:
        net: 모델. 단순히 클래스 이름을 받아올 목적.
        cost: 비용 리스트
    """

    # 주어진 네트워크 모델 인스턴스의 클래스 이름 저장
    net_type = net.name

    plt.figure(figsize=(5, 4))
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Loss Average")
    plt.title(f"{net_type} loss")

    path_ = GRAPH / f"{net_type}_loss.png"
    plt.savefig(path_)

    logger.info(f"Saved {net_type} loss figure to '{path_.as_posix()}'")


def draw_accuracy(*net_accuracy_pair: Tuple[Network, float]):
    """주어진 모델들과 정확도들을 하나의 그래프로 그려서 저장.

    Args:
        *net_accuracy_pair: (모델, 정확도) 튜플들
    """

    # 네트워크 이름과 정확도들을 분리
    net_names, accuracies = zip(*net_accuracy_pair)

    # 색 지정
    colors = sns.color_palette("Pastel1", len(net_accuracy_pair))

    # 0~1 범위로 정확도 제한
    axis = plt.gca()
    axis.set_ylim([0, 1])

    plt.figure(figsize=(7, 4))
    plt.bar(net_names, accuracies, color=colors)
    plt.ylabel("Accuracy")

    path_ = GRAPH / f"accuracy.png"
    plt.savefig(path_)

    logger.info(f"Saved combined accuracy figure to '{path_.as_posix()}'")
