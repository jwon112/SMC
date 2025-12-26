"""
Multi-label 데이터 분석 스크립트
"""
import pandas as pd
import os

# 데이터 로드
train_df = pd.read_csv('SMC/data/train/train.csv')
val_df = pd.read_csv('SMC/data/val/val.csv')
test_df = pd.read_csv('SMC/data/test/test.csv')

# 모든 데이터 합치기
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

label_cols = ['BN', 'CN', 'EF', 'NI']

# 라벨 개수 계산
all_df['num_labels'] = all_df[label_cols].sum(axis=1)

# 통계
total = len(all_df)
single_label = len(all_df[all_df['num_labels'] == 1])
multi_label = len(all_df[all_df['num_labels'] > 1])
zero_label = len(all_df[all_df['num_labels'] == 0])

print("="*60)
print("Multi-label 데이터 분석 결과")
print("="*60)
print(f"전체 샘플: {total}개")
print(f"Single-label: {single_label}개 ({single_label/total*100:.2f}%)")
print(f"Multi-label: {multi_label}개 ({multi_label/total*100:.2f}%)")
print(f"Zero-label: {zero_label}개 ({zero_label/total*100:.2f}%)")
print()

# Multi-label 샘플 분석
multi_label_df = all_df[all_df['num_labels'] > 1]

print("Multi-label 조합 분석:")
print("-"*60)

# 조합별 통계
combos = {}
for idx, row in multi_label_df.iterrows():
    active_labels = [label_cols[i] for i, val in enumerate([row['BN'], row['CN'], row['EF'], row['NI']]) if val == 1]
    combo = '+'.join(active_labels)
    combos[combo] = combos.get(combo, 0) + 1

print(f"총 {len(multi_label_df)}개의 Multi-label 샘플")
print(f"총 {len(combos)}가지 조합")
print()
print("조합별 통계 (빈도순):")
for combo, count in sorted(combos.items(), key=lambda x: x[1], reverse=True):
    print(f"  {combo}: {count}개 ({count/len(multi_label_df)*100:.1f}%)")

print()
print("Multi-label 샘플 예시 (처음 10개):")
print("-"*60)
for idx, row in multi_label_df.head(10).iterrows():
    active_labels = [label_cols[i] for i, val in enumerate([row['BN'], row['CN'], row['EF'], row['NI']]) if val == 1]
    print(f"{row['image']}: {' + '.join(active_labels)}")

# Split별 통계
print()
print("="*60)
print("Split별 Multi-label 비율:")
print("-"*60)
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    df['num_labels'] = df[label_cols].sum(axis=1)
    total_split = len(df)
    multi_split = len(df[df['num_labels'] > 1])
    print(f"{name}: {multi_split}/{total_split} ({multi_split/total_split*100:.2f}%)")

