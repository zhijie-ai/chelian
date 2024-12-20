# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/1/14 14:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
from networkx.algorithms.flow import shortest_augmenting_path
import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
G.add_edge('x','a', capacity=3.0)
G.add_edge('x','b', capacity=1.0)
G.add_edge('a','c', capacity=3.0)
G.add_edge('b','c', capacity=5.0)
G.add_edge('b','d', capacity=4.0)
G.add_edge('d','e', capacity=2.0)
G.add_edge('c','y', capacity=2.0)
G.add_edge('e','y', capacity=3.0)

# 将参数画到图上
# pos = nx.spring_layout(G)
# capacity = nx.get_edge_attributes(G, 'capacity')
# print(capacity)
# nx.draw(G, pos)
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G,pos, capacity)
# nx.draw_networkx(G)

# # 最大流
# flow_value = nx.maximum_flow_value(G, 'x', 'y', flow_func=shortest_augmenting_path)
# # print(flow_value)
#
# # 最小成本流
# G = nx.DiGraph()
# G.add_node('a', demand=-5)
# G.add_node('d', demand = 5)
# G.add_edge('a', 'b', weight = 3, capacity = 4)
# G.add_edge('a', 'c', weight = 6, capacity = 10)
# G.add_edge('b', 'd', weight = 1, capacity = 9)
# G.add_edge('c', 'd', weight = 2, capacity = 5)
# flow_cost, flow_dict = nx.capacity_scaling(G)
# print(flow_cost, flow_dict)

# 一个图G,一条路径经过图G的每一条边,且仅经过一次,这条路径称为欧拉路径．如果起点和终点同一点，则为欧拉回路
# 无向图：每个顶点的度数都是偶数则存在欧拉回路
# 有向图：每个顶点的入度都等于出度则存在欧拉回路
# DG = nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]})
# G = nx.Graph({0: [1,2], 1: [0,2], 2: [0,1,3,4], 3: [2,4], 4:[2,3]})
# print(nx.is_eulerian(DG))
# print(nx.is_eulerian(G))
# print(list(nx.eulerian_circuit(DG)))
# print(list(nx.eulerian_circuit(G)))


# # 最小点割集
# node_cut = nx.minimum_node_cut(G, flow_func=shortest_augmenting_path)
# pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# nx.draw_networkx(G, pos)
# plt.show()

# 对于带权无向图边切割，得到最小切割权之和，以及两个分离区域
# G = nx.Graph()
# G.add_edge('x','a', weight=3)
# G.add_edge('x','b', weight=1)
# G.add_edge('a','c', weight=3)
# G.add_edge('b','c', weight=5)
# G.add_edge('b','d', weight=4)
# G.add_edge('d','e', weight=2)
# G.add_edge('c','y', weight=2)
# G.add_edge('e','y', weight=3)
#
# cut_value, partition = nx.stoer_wagner(G)
# print(cut_value, partition)

# # 最大权重匹配，匈牙利，KM算法
# G = nx.Graph()
# G.add_weighted_edges_from([('A', 'a', 3), ('A', 'c', 4),
#                            ('B', 'a', 2), ('B', 'b', 1),
#                            ('B', 'c', 3), ('C', 'c', 5)])
# print(nx.max_weight_matching(G))

# 拓扑排序
G = nx.DiGraph()
G.add_edge('x','a', weight=3)
G.add_edge('a','c', weight=3)
G.add_edge('b','c', weight=5)
G.add_edge('b','d', weight=4)
G.add_edge('d','e', weight=2)
G.add_edge('c','y', weight=2)
G.add_edge('e','y', weight=3)
print(list(nx.topological_sort(G)))

# # 最小成本最大流
# G = nx.DiGraph()
# G.add_edge('a', 'b', weight = 3, capacity = 4)
# G.add_edge('a', 'c', weight = 6, capacity = 10)
# G.add_edge('b', 'd', weight = 1, capacity = 9)
# G.add_edge('c', 'd', weight = 2, capacity = 5)
# print(nx.max_flow_min_cost(G, 'a', 'd'))

#复杂网络生成
# ER随机图   # 随机生成20个节点，节点间的连接概率都是0.2
# ER = nx.random_graphs.erdos_renyi_graph(20, 0.2)
# pos = nx.shell_layout(ER)
# nx.draw(ER, pos, with_labels=False,edge_color='b', alpha=0.3, node_size=30)
# plt.show()

# WS小世界网络  生成一个含有n个节点、每个节点有k个邻居、以概率p随机化重连边的WS小世界网络。
# WS = nx.random_graphs.watts_strogatz_graph(20, 4, 0.3)
# pos = nx.circular_layout(WS)
# nx.draw(WS, pos, with_labels=False, node_size=30, edge_color='b', alpha=0.3)
# plt.show()

# # BA无标度网络 生成一个含有n个节点、每次加入m条边的BA无标度网络
# BA = nx.random_graphs.barabasi_albert_graph(10,2)
# pos = nx.spring_layout(BA)
# nx.draw(BA, pos, with_labels=False, node_size=30, edge_color='b', alpha=0.3)
# plt.show()

# 扩展ＢＡ无标度网络  节点数，添加新边数，两点间添加边的概率，边重连的概率
ExBA = nx.random_graphs.extended_barabasi_albert_graph(200, 4, 0.4, 0.2)
pos = nx.spring_layout(ExBA)
nx.draw(ExBA, pos, with_labels=False, node_size=30, edge_color='b', alpha=0.3)
plt.show()

