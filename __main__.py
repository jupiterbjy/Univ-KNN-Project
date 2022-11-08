"""
Fallback ML project

Titanic survivor prediction project
"""

from knn_class import KNN
from dataset import *


def main():
    k = 7
    network = KNN(k, train_data, train_label)

    print(f"{k}-nn Network test start")

    # 생존자 / 사망자 수 카운터
    total = [0, 0]

    # Dataframe 을 Iteration - 첫 값으로 인덱스가 추가되니 별도로 받음
    for passenger_id, *record in test_data:

        # 가중치를 고려한 다수결을 사용하여 예측
        result = network.majority_vote_weighted(record)

        # 카운터 +1
        total[result] += 1

        # 결과 출력. 이때 Numpy 배열을 사용하였으므로 승선자 ID도 float형이 되있으니 형변환 + 자릿수 4자리로 고정
        # 반환값이 1이면 생존, 0이면 사망 출력
        print(f"Passenger {int(passenger_id):4} estimated as {'survived' if result else 'deceased'}")

    # 예측된 생존자 / 사망자 수 출력
    print(f"Total estimated survivor: {total[0]} / Deceased: {total[1]}")


if __name__ == '__main__':
    main()
