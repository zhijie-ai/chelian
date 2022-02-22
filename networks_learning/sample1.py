import networkx as nx
import matplotlib.pyplot as plt

oo = float('inf')

# 创建无向图
# G = nx.Graph()
# G.add_node(1) # 添加节点１
# G.add_edge(2,3) #　添加节点２，３并链接２３节点
# print(G.nodes, G.edges, G.number_of_nodes(), G.number_of_edges())
# nx.draw(G)
# # plt.show()
#
# # 创建有向图
# G = nx.DiGraph()
# G.add_edge(2, 3)
# G.add_edge(3, 2)
# # G.to_undirected()  # 转换成无向图
# print(G.edges)
# nx.draw(G)
# plt.show()

# G = nx.DiGraph()
# G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, .5)])  # 给０１边加权３，　１２边加权７．５
# print(G.get_edge_data(1,2))
#
# G.add_weighted_edges_from([(2,3,5)], weight='color')
# G._node[1]['size'] = 10
# # G.nodes[1]['size'] = 10
# print(G.edges.data())
# print(G.nodes.data())
# nx.draw(G)
# plt.show()

g_data = [(1, 2, 6), (1, 3, 1), (1, 4, 5),
          (2, 3, 5),  (2, 5, 3),
          (3, 4, 5), (3, 5, 6), (3, 6, 4), (4, 6, 2),
          (5, 6, 6)]
# g = nx.Graph()
# g.add_weighted_edges_from(g_data)
# tree = nx.minimum_spanning_tree(g, algorithm='prim')
# print(tree.edges(data=True))

#最短路劲
# G = nx.path_graph(5) # 0-1-2-3-4链
# nx.draw(G)
# plt.show()
# print(nx.dijkstra_path(G,0,4))

# 所有节点之间的最短路劲
# G = nx.Graph()
# G.add_weighted_edges_from(g_data)
# gen = nx.all_pairs_shortest_path(G)
# print(list(gen))

# 各点之间的可达性
# G = nx.Graph()
# G.add_weighted_edges_from(g_data)
# print(nx.communicability(G))

# 获取图中非连通点的列表
# G = nx.Graph()
# G.add_edge(1, 2)
# G.add_node(3)
# print(list(nx.isolates(G)))

# 遍历
G= nx.Graph()
G.add_weighted_edges_from(g_data)
d_gen = nx.dfs_edges(G,1) # 按边深度搜索，1为起点
b_gen = nx.dfs_edges(G, 1)
print(list(d_gen), list(b_gen))
print(nx.dfs_tree(G,1).nodes())# 按点深搜