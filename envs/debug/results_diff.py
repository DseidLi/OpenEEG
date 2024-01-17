import argparse

import pandas as pd


def compare_csv(file1, file2):
    # 读取 CSV 文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 合并两个 DataFrame 并找出不同
    merged = pd.merge(df1,
                      df2,
                      on='EpochID',
                      how='outer',
                      suffixes=('_file1', '_file2'))
    differences = merged[
        merged['Prediction_file1'] != merged['Prediction_file2']]

    # 计算不同的行数
    diff_count = len(differences)

    # 如果不同的行数超过 10 行，则打印前 10 行的差异
    if diff_count > 10:
        print('Differences in the first 10 rows:')
        print(differences.head(10))
    else:
        print('All differences:')
        print(differences)

    return diff_count


def main():
    parser = argparse.ArgumentParser(description='Compare two CSV files.')
    parser.add_argument('file1', type=str, help='Path to the first CSV file')
    parser.add_argument('file2', type=str, help='Path to the second CSV file')

    args = parser.parse_args()

    difference_count = compare_csv(args.file1, args.file2)
    print('Total number of different rows:', difference_count)


if __name__ == '__main__':
    main()
