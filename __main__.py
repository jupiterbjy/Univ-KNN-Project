"""
Fallback ML project

Titanic survivor prediction project
"""

import multiprocessing as mp

from loguru import logger
from Algorithms import knn, dnn, logistic_regression, linear_regression, draw_accuracy
from dataset import *


def main():
    # 데이터 정보 출력
    logger.info(
        f"Training data set #: {len(train_x)} / Test data set #: {len(test_x)}"
    )

    t = dnn.test(train_x, train_y, test_x, test_y, 3, 7, 100, 0.01)

    # 테스트 목록 작성 [(함수, (인자들)), ...}
    tests = [
        (knn.test, (train_x, train_y, test_x, test_y, 1)),
        (knn.test, (train_x, train_y, test_x, test_y, 7)),
        (linear_regression.test, (train_x, train_y, test_x, test_y, 1000, 0.1)),
        (logistic_regression.test, (train_x, train_y, test_x, test_y, 1000, 0.1)),
    ]

    # 병렬로 실행. 시간 절약.
    # 이경우 디버깅 시 pool 종료 후 어느 중단점에서건 오류 발생. Pycharm 버그.
    # https://youtrack.jetbrains.com/issue/PY-54447
    with mp.Pool(processes=4) as pool:
        processes = [pool.apply_async(test, arg) for (test, arg) in tests]
        results = [p.get() for p in processes]

    # 정확도 종합
    draw_accuracy(*results)


if __name__ == "__main__":
    main()
