"""
Fallback ML project

Titanic survivor prediction project
"""

import multiprocessing as mp

from loguru import logger

from Algorithms import knn, logistic_regression, dnn
from dataset import *


def main():
    # 데이터 정보 출력
    logger.info(f"Data preprocessing done - Train set: {len(train_x)} / Test set: {len(test_x)}")

    # KNN 테스트
    res_1nn = mp.Process(target=knn.test, args=(train_x, train_y, test_x, test_y, 1))
    res_7nn = mp.Process(target=knn.test, args=(train_x, train_y, test_x, test_y, 7))

    # Logistic Regression 테스트
    res_ll = mp.Process(target=logistic_regression.test, args=(train_x, train_y, test_x, test_y, 1000, 0.001))

    processes = [res_1nn, res_7nn, res_ll]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    # DNN 테스트
    print("b")


if __name__ == '__main__':
    main()
