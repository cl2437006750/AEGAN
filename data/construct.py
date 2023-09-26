import os
import numpy as np
import scipy.sparse as sp

from torch_geometric.datasets import Planetoid

# 加载数据集
dataset = Planetoid(root='../data/Cora', name='Cora')
data = dataset[0]

tensor = data.edge_index
print(data.edge_index)
# 将 tensor 中的数据按列转换成列表
data = tensor.T.tolist()

# 将列表写入文件中
# with open('../data/cora_graph.txt', 'w') as f:
#     for row in data:
#         row_str = ' '.join([str(x) for x in row]) + '\n'
#         f.write(row_str)

# print(data.x)
# np.savetxt('data/pubmed.txt', data.x.numpy())

# print(data.y)
# tensor = data.y
# array = tensor.numpy()
#
# # 将数组保存到文件中
# np.savetxt('data/Cora.txt', array, fmt='%d')