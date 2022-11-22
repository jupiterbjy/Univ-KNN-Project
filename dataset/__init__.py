"""
Titanic Dataset library

Data sourced from:
https://www.kaggle.com/competitions/titanic/data


"""

from typing import Tuple, TypedDict
import pathlib

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).parent
CSV_FILE = ROOT / "survivor_data.csv"

# 문자열 값을 정수로 번역할 때 쓸 Dictionary
GENDER_TRANS = {"male": 0, "female": 1}
EMBARK_TRANS = {"Q": 0, "S": 1, "C": 2}


# wildcard import 시 불러올 대상 제한
__all__ = ["train_y", "train_x", "test_x", "test_y"]


class Record(TypedDict):
    # 데이터 컬럼 형식을 적어둔 참고용 클래스. 실제로 쓰진 않음.
    # 나머지 데이터 들은 널값이 너무 많거나 수치화 할 수 없어 제거.

    PassengerId: int
    # 승선자 ID

    Pclass: int
    # 등급, 1 2 3등급 = 1, 2, 3

    Sex: int
    # 남, 녀 = 0, 1

    Age: float
    # 나이. 정확한 나이를 알수 없는 승객의 경우 소수점이 존재.

    SibSp: int
    # 타이타닉에 동승한 형제 / 배우자 수

    Parch: int
    # 타이타닉에 동승한 부모 / 자식 수

    Fare: float
    # 요금

    Embarked: int
    # 탑승 항구, Cherbourg, Queenstown, Southampton = C, Q, S = 0, 1, 2


def preprocess(train_ratio=0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    학습 데이터 / 테스트 데이터 전처리 함수.
    여기서는 승선자 ID가 필요 없으므로 제거함에 유의.
    """

    # 파일 읽기 & 널값 레코드 제거
    data_raw = pd.read_csv(CSV_FILE).dropna()

    # 신경망 으로 처리할 수 있게 값들을 전부 수치화
    data_raw["sex"] = data_raw["sex"].map(GENDER_TRANS)
    data_raw["embarked"] = data_raw["embarked"].map(EMBARK_TRANS)

    # 최대-최소 값으로 정규화
    scaled = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min())

    # 학습 데이터, 테스트 데이터 분할
    sep = int(train_ratio * len(scaled))
    train = scaled[:sep]
    test = scaled[sep:]

    # 각각 생존자 컬럼을 분리
    train_l = train["survived"]
    test_l = test["survived"]

    # 승선자 ID, 생존자 컬럼을 삭제
    train = train.drop(["survived"], axis=1)
    test = test.drop(["survived"], axis=1)

    return train_l.to_numpy(), train.to_numpy(), test_l.to_numpy(), test.to_numpy()


train_y, train_x, test_y, test_x = preprocess()
