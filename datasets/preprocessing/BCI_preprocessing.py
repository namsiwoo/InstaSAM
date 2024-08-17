import h5py

# 읽기 모드로 파일 열기
with h5py.File('/media/NAS/nas_70/open_dataset/BCI_dataset/BCData/annotations/test/positive/0.h5', 'r') as f:
    # 데이터셋 로드
    print(list(f.keys()))

    # data = f['my_dataset'][:]
    # print(data)