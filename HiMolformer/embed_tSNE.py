import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

epoch_interval = 1

# CSV 파일 로드
HiMol_df = pd.read_csv('embeddings/HiMol_embeddings.csv')
Molformer_df = pd.read_csv('embeddings/Molformer_embeddings.csv')
HiMolformer_df = pd.read_csv('embeddings/HiMolformer_embeddings.csv')

# 지정된 에포크 주기에 있는 에포크 번호를 고유하게 가져옵니다.
unique_epochs = HiMol_df['epoch'].unique()
selected_epochs = [epoch for epoch in unique_epochs if epoch % epoch_interval == 0]

# 각 선택된 에포크에 대해 시각화 수행
for epoch in selected_epochs:
    # 해당 에포크의 데이터 필터링
    HiMol_embeddings = HiMol_df[HiMol_df['epoch'] == epoch].drop('epoch', axis=1).values
    Molformer_embeddings = Molformer_df[Molformer_df['epoch'] == epoch].drop('epoch', axis=1).values
    HiMolformer_embeddings = HiMolformer_df[HiMolformer_df['epoch'] == epoch].drop('epoch', axis=1).values

    # PCA 적용하여 데이터 축소
    pca = PCA(n_components=50)  # 임의의 컴포넌트 수, 예: 50
    HiMol_pca = pca.fit_transform(HiMol_embeddings)
    Molformer_pca = pca.fit_transform(Molformer_embeddings)
    HiMolformer_pca = pca.fit_transform(HiMolformer_embeddings)
    tsne = TSNE(n_components=2, random_state=0)
    HiMol_tsne = tsne.fit_transform(HiMol_pca)
    Molformer_tsne = tsne.fit_transform(Molformer_pca)
    HiMolformer_tsne = tsne.fit_transform(HiMolformer_pca)

    # tsne = TSNE(n_components=2, random_state=0)
    # HiMol_tsne = tsne.fit_transform(HiMol_embeddings)
    # Molformer_tsne = tsne.fit_transform(Molformer_embeddings)
    # HiMolformer_tsne = tsne.fit_transform(HiMolformer_embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(HiMol_tsne[:, 0], HiMol_tsne[:, 1], c='blue', label='HiMol')
    plt.scatter(Molformer_tsne[:, 0], Molformer_tsne[:, 1], c='red', label='Molformer')
    plt.scatter(HiMolformer_tsne[:, 0], HiMolformer_tsne[:, 1], c='green', label='HiMolformer')
    plt.legend(fontsize=30)

    plt.title(f'MLM Avg768 Epoch {epoch}', fontsize=30, pad=20)

    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'embeddings/PCA_embeddings_epoch_{epoch}.png', bbox_inches='tight')
    plt.close()
