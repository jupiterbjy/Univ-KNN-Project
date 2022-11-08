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
CSV_TRAIN = ROOT / "train.csv"
CSV_TEST = ROOT / "test.csv"

# 문자열 값을 정수로 번역할 때 쓸 Dictionary
GENDER_TRANS = {"male": 0, "female": 1}
EMBARK_TRANS = {"Q": 0, "S": 1, "C": 2}


# wildcard import 시 불러올 대상 제한
__all__ = ["train_label", "train_data", "test_data"]


class Record(TypedDict):
    # 데이터 컬럼 형식을 적어둔 참고용 클래스. 실제로 쓰진 않음.

    PassengerId: int
    # 승선자 ID

    Pclass: int
    # 등급, 1 2 3등급 = 1, 2, 3

    Sex: int
    # 남, 녀 = 0, 1

    Age: float
    # 나이. 정확한 나이를 알수 없는 승객의 경우 소수점 존재.

    SibSp: int
    # 타이타닉에 동승한 형제 / 배우자 수

    Parch: int
    # 타이타닉에 동승한 부모 / 자식 수

    Fare: float
    # 요금

    Embarked: int
    # 탑승 항구, Cherbourg, Queenstown, Southampton = C, Q, S = 0, 1, 2


def preprocess_test() -> np.ndarray:
    """
    테스트 데이터 전처리 함수 - 여기서는 누가 생존했는지 확인하기 위해 승선자 ID를 보존.
    """

    # 파일 읽기 & 널값 레코드 제거
    data_raw = pd.read_csv(CSV_TEST).dropna()

    # KNN 으로 처리할 수 있게 값들을 전부 수치화
    data_raw["Sex"] = data_raw["Sex"].map(GENDER_TRANS)
    data_raw["Embarked"] = data_raw["Embarked"].map(EMBARK_TRANS)

    return data_raw.to_numpy()


def preprocess_train() -> Tuple[np.ndarray, np.ndarray]:
    """
    학습 데이터 전처리 함수 - 여기서는 승선자 ID가 필요 없으므로 제거함에 유의.
    """

    # CSV 파일서 쓰지 않을 필드는 사전에 제거

    # 파일 읽기 & 널값 레코드 제거
    data_raw = pd.read_csv(CSV_TRAIN).dropna()

    # KNN 으로 처리할 수 있게 값들을 전부 수치화
    data_raw["Sex"] = data_raw["Sex"].map(GENDER_TRANS)
    data_raw["Embarked"] = data_raw["Embarked"].map(EMBARK_TRANS)

    # 생존자 컬럼을 분리
    label = data_raw["Survived"]

    # 승선자 ID, 생존자 컬럼을 삭제
    data = data_raw.drop(["PassengerId", "Survived"], axis=1)

    return label.to_numpy(), data.to_numpy()


train_label, train_data = preprocess_train()
test_data = preprocess_test()
