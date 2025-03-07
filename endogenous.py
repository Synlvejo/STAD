import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import re
import csv
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import numpy as np
import sys
# from networkx.drawing.nx_agraph import graphviz_layout
# import pygraphviz
# # import graphviz as pgv


def read_csv_file(filepath, delimiter=',', quotechar='"', has_header=True):
    """
    读取 CSV 文件的每一行，并返回数据。
    
    :param filepath: CSV 文件的路径
    :param delimiter: 字段分隔符，默认为 ','
    :param quotechar: 引号字符，默认为 '"'
    :param has_header: 是否包含标题行，默认为 True
    :return: 如果有标题行，返回字典列表；否则，返回列表列表
    """
    data = []
    
    with open(filepath, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=delimiter, quotechar=quotechar)
        
        # 如果文件有标题行
        if has_header:
            headers = next(reader)
            for row in reader:
                if row:  # 跳过空行
                    data.append(dict(zip(headers, row)))
        else:
            for row in reader:
                if row:  # 跳过空行
                    data.append(row)
    
    return data

def reverse_and_join(lst):
    # 反转列表
    reversed_lst = lst[::-1]
    
    # 将所有元素转换为字符串
    str_list = [str(item) for item in reversed_lst]
    
    # 用 '/' 连接所有字符串
    result = '/'.join(str_list)
    
    return result

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.weight = 0  # 初始化权重为1

    def add_child(self, child_node):
        self.children[child_node.name] = child_node
        self.weight += child_node.weight  # 更新当前节点的权重

    def __repr__(self, level=0):
        ret = "\t" * level + f"{self.name} (Weight: {self.weight})\n"
        for child in self.children.values():
            ret += child.__repr__(level + 1)
        return ret

# def build_tree(lines):
#     root = TreeNode("/")

#     # 第一遍遍历：构建树的基本结构，设置初步的权重
#     for line in lines:
#         weight = int(line['score'])
#         backtrace = line['backtrace']
#         node = line['node']
#         parts = backtrace.strip().split("\n")
#         parts = [item[:-1] for item in parts[:-1]] + [parts[-1]]
#         realbacktrace = []
#         realbacktrace.append(parts[0].split(":")[-3].split("[")[0].split(" ")[0])
#         current_node = root
#         root.weight = 0

#         # 生成节点
#         for part in parts[1:]:
#             file_line = part.split("/")[-1].split("]")[0]
#             realbacktrace.append(file_line) 
#         realbacktrace = reverse_and_join(realbacktrace).split("/")
#         # print(realbacktrace)
#         realbacktrace[1] = realbacktrace[1].split(")")[0]+")"
#         realbacktrace[0] = realbacktrace[0]+"]"

#         #建树
#         for graph_node in realbacktrace:  # Skip the first empty part
#             if graph_node not in current_node.children:
#                 # 节点不存在，创建新节点
#                 current_node.add_child(TreeNode(graph_node))
#                 current_node = current_node.children[graph_node]
#                 current_node.weight = weight
#             else:
#                 # 节点已存在，直接获取
#                 current_node = current_node.children[graph_node]
#                 current_node.weight += weight  # 为叶子节点累加权重
    
#     return root

def build_tree_normal(lines):
    root = TreeNode("/")

    # 第一遍遍历：构建树的基本结构，设置初步的权重
    for line in lines:
        weight = int(line['score'])
        backtrace = line['backtrace']
        node = line['node']
        parts = backtrace.strip().split("\n")
        parts = [item[:-1] for item in parts[:-1]] + [parts[-1]]
        realbacktrace = []
        realbacktrace.append(parts[0].split(":")[-3].split("[")[0].split(" ")[0])
        current_node = root
        root.weight = 0

        # 生成节点
        for part in parts[1:]:
            file_line = part.split("/")[-1].split("]")[0]
            realbacktrace.append(file_line) 
        realbacktrace = reverse_and_join(realbacktrace).split("/")
        # print(realbacktrace)
        realbacktrace[1] = realbacktrace[1].split(")")[0]+")"
        realbacktrace[0] = realbacktrace[0].split(":")[0].split(")")[0] + ")"

        # 建树
        for graph_node in realbacktrace:  # Skip the first empty part
            if graph_node not in current_node.children:
                # 节点不存在，创建新节点
                current_node.add_child(TreeNode(graph_node))
                current_node = current_node.children[graph_node]
                current_node.weight = weight
            else:
                # 节点已存在，直接获取
                current_node = current_node.children[graph_node]
                current_node.weight += weight  # 为叶子节点累加权重

    # 获取所有节点的权重
    all_weights = []

    def collect_weights(node):
        all_weights.append(node.weight)
        for child in node.children.values():
            collect_weights(child)
    
    collect_weights(root)

    # 计算权重的最小值和最大值
    min_weight = min(all_weights)
    max_weight = max(all_weights)

    # 定义归一化函数，将权重归一化到 0-100
    def normalize_weight(weight):
        if max_weight == min_weight:
            return 0  # 如果所有权重相同，则归一化结果均为 0
        return 100 * (weight - min_weight) / (max_weight - min_weight)
    
    # 归一化所有节点的权重
    def normalize_tree_weights(node):
        node.weight = normalize_weight(node.weight)
        for child in node.children.values():
            normalize_tree_weights(child)
    
    normalize_tree_weights(root)

    return root

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()
    
def read_csv_file(filepath, delimiter=',', quotechar='"', has_header=True):
    """
    读取 CSV 文件的每一行，并返回数据。
    
    :param filepath: CSV 文件的路径
    :param delimiter: 字段分隔符，默认为 ','
    :param quotechar: 引号字符，默认为 '"'
    :param has_header: 是否包含标题行，默认为 True
    :return: 如果有标题行，返回字典列表；否则，返回列表列表
    """
    data = []
    
    with open(filepath, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=delimiter, quotechar=quotechar)
        
        # 如果文件有标题行
        if has_header:
            headers = next(reader)
            for row in reader:
                if row:  # 跳过空行
                    data.append(dict(zip(headers, row)))
        else:
            for row in reader:
                if row:  # 跳过空行
                    data.append(row)
    
    return data

def print_tree(root):
    print(root)

def add_edges(graph, node, parent=None):
    if node is None:
        return
    
    # 如果节点不在图中，则添加节点并设置初始权重
    if node.name not in graph:
        graph.add_node(node.name, weight=node.weight)
    else:
        # 如果节点已经在图中，则更新其权重
        current_weight = graph.nodes[node.name].get('weight', 0)
        graph.nodes[node.name]['weight'] = current_weight + node.weight
    
    # 如果有父节点，添加边
    if parent:
        if parent.name not in graph:
            graph.add_node(parent.name, weight=parent.weight)
        # 确保父节点的权重也被处理
        parent_weight = graph.nodes[parent.name].get('weight', 0)
        graph.nodes[parent.name]['weight'] = parent_weight
        graph.add_edge(parent.name, node.name)
    
    # 递归处理子节点
    for child in node.children.values():
        add_edges(graph, child, node)


def get_node_depths(graph):
    depths = {}
    for node in nx.topological_sort(graph):
        ancestors = nx.ancestors(graph, node)
        if ancestors:
            depths[node] = max(depths[ancestor] for ancestor in ancestors) + 1
        else:
            depths[node] = 0
    return depths


def spread_nodes_at_same_level(pos, node_depths, node_weights):
    """ Spread nodes at the same depth level horizontally with offset, sorted by weight. """
    levels = list(set(node_depths.values()))
    for level in levels:
        nodes_at_level = [(node, node_weights[node]) for node in node_depths if node_depths[node] == level]
        # Sort nodes by weight in descending order
        nodes_at_level.sort(key=lambda x: x[1], reverse=True)
        num_nodes = len(nodes_at_level)
        x_coords = np.linspace(-0.5, 0.5, num_nodes)
        for i, (node, weight) in enumerate(nodes_at_level):
            pos[node] = (x_coords[i], pos[node][1])
    return pos

def aggregate_node_weights(root):
    # 创建一个用于累加权重的字典
    weight_accumulator = {}

    # 遍历所有节点
    for node in root:
        # 获取当前节点的权重
        weight = root.nodes[node].get('weight', 0)
        #print(node, weight)
        
        # 累加权重到对应的节点名字
        if node in weight_accumulator:
            weight_accumulator[node] += weight
        else:
            weight_accumulator[node] = weight
    
    # 将累加结果作为最终的 node_weights
    node_weights = weight_accumulator
    return node_weights

def visualize_tree(root, deep_depth):
    print(root)
    graph = nx.DiGraph()
    add_edges(graph, root)
    # print(graph.nodes)

    # Access node weights directly from TreeNode
    # print(root)
    node_weights = {node: graph.nodes[node].get('weight', 0) for node in graph.nodes}
    # node_weights = aggregate_node_weights(graph)
    print (node_weights)

    # Get node depths
    node_depths = get_node_depths(graph)

    # Filter out nodes with depth 0
    nodes_to_include = [node for node in graph.nodes if node_depths[node] != 0]
    filtered_graph = graph.subgraph(nodes_to_include).copy()
    filtered_node_weights = {node: node_weights[node] for node in nodes_to_include}
    filtered_node_depths = {node: node_depths[node] for node in nodes_to_include}
    print(filtered_node_weights)

    # Normalize weights for color mapping
    weights = list(filtered_node_weights.values())
    min_weight = min(weights)
    max_weight = max(weights)

    # Define a colormap and normalization
    cmap = plt.get_cmap('Oranges')
    # norm = Normalize(vmin=min_weight, vmax=max_weight)
    norm = LogNorm(vmin=min_weight + 1e-5, vmax=max_weight)

    def weight_to_color(weight):
        return cmap(norm(weight)-0.2)

    # Create a layout for the nodes
    pos = nx.spring_layout(filtered_graph, seed=42)

    # Adjust positions to ensure nodes at the same depth are on the same horizontal line and layers alternate
    pos = {node: (coord[0], -filtered_node_depths[node]) for node, coord in pos.items()}
    pos = spread_nodes_at_same_level(pos, filtered_node_depths, filtered_node_weights)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 6))  # Adjust size as needed

    # Draw the graph without labels
    node_colors = [weight_to_color(filtered_node_weights[node]) for node in filtered_graph.nodes()]
    print_tree(graph)
    nx.draw(filtered_graph, pos, with_labels=False, arrows=True, node_size=400, node_color=node_colors, edge_color='lightgray', ax=ax)

    # Draw node labels separately with matplotlib for rotation and alignment
    labels = {node: node for node in filtered_graph.nodes()}
    for node, (x, y) in pos.items():
        label = labels[node]
        # Check if the node's depth is between 0 and 5, inclusive
        if 0 <= filtered_node_depths[node] <= deep_depth:
            # For nodes at depth 0-5, set the label to be horizontal
            ax.text(x, y, label, fontsize=8, fontweight='bold', ha='center', va='center', rotation=0)
        else:
            # For other nodes, keep the label vertical
            ax.text(x, y, label, fontsize=7, fontweight='bold', ha='center', va='center', rotation=15)

    # Adjust layout to make room for the colorbar
    plt.subplots_adjust(right=0.85)  # Adjust this value to control the spacing

    # Create a colorbar with the correct colormap and normalization
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, label='Anomaly Score')

    # Save the figure with tight bounding box to remove extra white space
    plt.savefig('../source/5-end.svg', bbox_inches='tight', pad_inches=0.1)

    plt.show()


def get_node_layers(root):
    layers = {}
    
    def dfs(node, depth):
        if depth not in layers:
            layers[depth] = []
        layers[depth].append(node.name)
        for child in node.children.values():
            dfs(child, depth + 1)
    
    dfs(root, 0)
    return [layers[i] for i in sorted(layers.keys())]



# Replace 'input.txt' with your actual input file path
# input_file_path = 'lammps_128_backtrace'
input_file_path = sys.argv[1]
deep_depth = int(sys.argv[2])
lines = read_csv_file(input_file_path)
# tree = build_tree(lines)
tree = build_tree_normal(lines)
# print_tree(tree)
visualize_tree(tree,deep_depth)



