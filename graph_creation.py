import os
import re
import copy
import torch
import timeit
import logging
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx

# Configure logging
logging.basicConfig(
    filename='graph_creation.log',
    filemode='a',
    format='%(levelname)s - %(asctime)s - %(name)s - %(funcName)s - %(lineno)s - %(message)s',
    level=logging.DEBUG
)

FolderName = ""
dynamic_bb_run = {}
cycles_list = []
simple_cycles_list = []
bridges = []
nn_vars = 1
edge_weight = {}
raw_location = "../../data/raw/train"
graph_export_location = "fd"
FI_result_location = "fd"


def check_existence(label, dot_files_list):
    """ ??? """
    ext = []
    if label:
        for dt in dot_files_list:
            if dt in label:
                ext.append(dt)
    return ext


def get_dot_files_name_list():
    """ this function returns list of this program dot files"""
    return os.listdir(raw_location + FolderName + '/dotfiles/')


def convert_dot_names_to_function_names(dot_names):
    """ each dot file is for one specific function
    this function returns each dot file corresponding function"""
    for i in range(len(dot_names)):
        dot_names[i] = dot_names[i][4:-4]
    return dot_names


def get_graphs_dict(dot_files_list):
    """ each dot file represent a control flow graph of
    function, we create dictionary that get this graph in networkx
    representation. key for each item is function name of dot file """
    dct = {}
    for i in range(len(dot_files_list)):
        dct[str(convert_dot_names_to_function_names([dot_files_list[i]])[0])] = nx.DiGraph(
            nx.nx_pydot.read_dot(
                raw_location + str(FolderName) + "/dotfiles/" + str(dot_files_list[i])))  # Read Graph
    return dct


def get_node_label(G, node):
    """
    each node in CFG graph is a basic block and has a name and some information
    in the case of they are list of instructions in the basic block in text format
    :param G: graph
    :param node: node we want its instruction
    :return: instructions of this node(Basic Block) in text format
    """
    for i in G.nodes.data("label"):
        if i[0] == node:
            return i[1]


def remove_ports(G):
    """ each node in a dot file has ports that are for example yes/no
    or True/False if we don't remove these ports in step we convert
    networkx graph to pytorch they become a node, se we remove them here """
    for x in G.nodes():
        if ":" in str(x):
            mapping = {x: str(x)[0:str(x).index(":")]}
            label = get_node_label(G, str(x)[0:str(x).index(":")])
            G = nx.relabel_nodes(G, mapping)
            nx.set_node_attributes(G, {str(x)[0:str(x).index(":")]: label}, name="label")
    return G


def dg_duplicate_node(G, node, name):
    """
    it gets a graph and add new node to it
    suppose we have a graph G with node v1 that has edges goes
    to v2,v3,v4. in this function we call(G, v1, "n1")
    then we have a new graph G that v1 gores to n1 and n1 goes to
    v2,v3,v4
    :param G: input graph
    :param node: node we want to duplicate, it's better to say node we
    want to add new node after it.
    :param name: new node name
    :return: new graph_export_location with new node
    """
    successors = [n for n in G.neighbors(node)]
    # print(get_node_label(G, node))
    features = {
        # 'label': "-12:just",
        # 'label': '"{\\"-12\\":\\l  spare-node}"',
        'label': get_node_label(G, node),
    }
    G.add_node(name, **features)

    for i in successors:
        G.add_edge(name, i)
    for i in successors:
        G.remove_edge(node, i)
    G.add_edge(node, name)
    return G


def add_graph_to_node(g_out, g_in, node_s, node_t):
    """
    we have a graph that represent our control flow graph.
    we call it g_out. in one of its nodes(node_s) it calls another
    control flow graph, we call it g_in. this function put graph g_in
     between nodes nodes_s and node_t. we know that each control
    flow graph starts with one node.
    :param g_out: main graph that will be inter-procedural control flow graph
    :param g_in: if we have a function call in one node of g_out the function
    will have a CFG, this is CFG of that function that we want to put it in main graph
    between nodes nodes_s and node_t
    :param node_s:
    :param node_t:
    :return: ICFG of graphs g_out and g_in
    """
    srcs = [x for x in g_in.nodes() if g_in.out_degree(x) >= 1 and g_in.in_degree(x) == 0]
    trgs = [x for x in g_in.nodes() if g_in.out_degree(x) == 0 and g_in.in_degree(x) >= 1]
    g_out.remove_edge(node_s, node_t)
    for i in srcs:
        g_out.add_edge(node_s, i)
    for i in trgs:
        g_out.add_edge(i, node_t)
    FC = nx.compose(g_out, g_in)
    return FC


def get_dyn_run_of_bb(bb):
    """
    :param bb: a basic block or node in graph
    :return: number of times it executed in dynamic execution
    """
    # print("bb:", bb)
    # print(dynamic_bb_run)
    return dynamic_bb_run.get(int(bb), 1)

    count = 0
    for k in dynamic_bb_run.keys():
        count += dynamic_bb_run[k]
    if bb in dynamic_bb_run.keys():
        return dynamic_bb_run[bb] / count
    else:
        return 0


def get_bb_sig(inp_str):
    """
    each basic block consist of some instructions
     feature matrix is generated for each node. The LLVM IR consists of 65 distinct instructions,
    which are categorized into 10 groups. For every transition from one instruction to another within
     a basic block the corresponding cell in the matrix is incremented by one.
    :param inp_str: list of instructions of basic block in string format
    :return: 10*10 matrix represent transition from each instruction group to another one
    """
    print("sssssssssssssssssssssss")
    instructions = inp_str.split("\l")
    num_features_category = 10
    change_mat = [[0] * num_features_category] * num_features_category
    list_mat = []
    helper = {"ret": 0, "br": 1, "switch": 2, "indirectbr": 3, "invoke": 4, "callbr": 5, "resume": 6, "catchswitch": 7,
              "catchret": 8, "cleanupret": 9, "unreachable": 10, "fneg": 11, "add": 12, "fadd": 13, "sub": 14,
              "fsub": 15, "mul": 16, "fmul": 17, "udiv": 18, "sdiv": 19, "fdiv": 20, "urem": 21, "srem": 22, "frem": 23,
              "shl": 24, "lshr": 25, "ashr": 26, "and": 27, "or": 28, "xor": 29, "extractelement": 30,
              "insertelement": 31, "shufflevector": 32, "extractvalue": 33, "insertvalue": 34, "alloca": 35, "load": 36,
              "store": 37, "fence": 38, "cmpxchg": 39, "atomicrmw": 40, "getelementptr": 41, "trunc": 42, "zext": 43,
              "sext": 44, "fptrunc": 45, "fpext": 46, "fptoui": 47, "fptosi": 48, "uitofp": 49, "sitofp": 50,
              "ptrtoint": 51, "inttoptr": 52, "bitcast": 53, "addrspacecast": 54, "icmp": 55, "fcmp": 56, "phi": 57,
              "select": 58, "freeze": 59, "call": 60, "va_arg": 61, "landingpad": 62, "catchpad": 63, "cleanuppad": 64}
    control_flow_range = range(0, 11)
    arithmetic_range = range(11, 24)
    logic_range = range(24, 30)
    vector_operations_range = range(30, 33)
    memory_operations_range = range(35, 42)
    type_conversions_range = range(42, 55)
    comparison_operations_range = range(55, 57)
    data_flow_range = range(57, 60)
    function_call_range = range(60, 62)
    exception_handling_range = range(62, 65)
    for instruction in instructions:
        lst = []
        inst_category = -1
        for key in helper:
            if key in instruction:
                i = helper[key]
                # print("i:", i)
                if i in control_flow_range:
                    inst_category = 0
                elif i in arithmetic_range:
                    inst_category = 1
                elif i in logic_range:
                    inst_category = 2
                elif i in vector_operations_range:
                    inst_category = 3
                elif i in memory_operations_range:
                    inst_category = 4
                elif i in type_conversions_range:
                    inst_category = 5
                elif i in comparison_operations_range:
                    inst_category = 6
                elif i in data_flow_range:
                    inst_category = 7
                elif i in function_call_range:
                    inst_category = 8
                elif i in exception_handling_range:
                    inst_category = 9
                lst.append(inst_category)
        if lst:
            list_mat.append(lst)
            lst = []

    feats = [[0.0] * 10] * 10
    for r in range(1, len(list_mat)):
        for c in range(len(list_mat[r])):
            g = list_mat[r - 1][-1]
            f = list_mat[r][c]
            feats[g][f] += 1
    feat = []
    for r in feats:
        feat.extend(r)

    feat1 = [0.0] * 10
    feat2 = [0.0] * 10
    # print(list_mat)
    for i in list_mat[:-1]:
        for k in i:
            feat1[k] += 1

    for i in list_mat[1:]:
        for k in i:
            feat2[k] -= 1
    feat1.extend(feat2)
    return feat


def to_feature_old(flist):
    """
    :param flist: get list of instructions that are called in this node of CFG(Basic block)
    then create a list of features that will be assigned to corresponding node of pyg graph node.
    :return:
    """
    str_list = str(flist[1])
    cat_feat = get_bb_sig(str_list)
    sub = "\""

    BB_name = str_list[str_list.find("{", 1) + 3:str_list.find(":", 2) - 2]
    # print(str_list)
    # print(BB_name, "::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    if BB_name:
        dyn = get_dyn_run_of_bb(int(BB_name))
    else:
        dyn = 0

    features = [0] * 65
    helper = {"ret": 0, "br": 1, "switch": 2, "indirectbr": 3, "invoke": 4, "callbr": 5, "resume": 6, "catchswitch": 7,
              "catchret": 8, "cleanupret": 9, "unreachable": 10, "fneg": 11, "add": 12, "fadd": 13, "sub": 14,
              "fsub": 15, "mul": 16, "fmul": 17, "udiv": 18, "sdiv": 19, "fdiv": 20, "urem": 21, "srem": 22, "frem": 23,
              "shl": 24, "lshr": 25, "ashr": 26, "and": 27, "or": 28, "xor": 29, "extractelement": 30,
              "insertelement": 31, "shufflevector": 32, "extractvalue": 33, "insertvalue": 34, "alloca": 35, "load": 36,
              "store": 37, "fence": 38, "cmpxchg": 39, "atomicrmw": 40, "getelementptr": 41, "trunc": 42, "zext": 43,
              "sext": 44, "fptrunc": 45, "fpext": 46, "fptoui": 47, "fptosi": 48, "uitofp": 49, "sitofp": 50,
              "ptrtoint": 51, "inttoptr": 52, "bitcast": 53, "addrspacecast": 54, "icmp": 55, "fcmp": 56, "phi": 57,
              "select": 58, "freeze": 59, "call": 60, "va_arg": 61, "landingpad": 62, "catchpad": 63, "cleanuppad": 64}

    for key in helper:
        features[helper[key]] += str(flist[1]).count(key)
    features[60] = str(flist[1]).count('call') - str(flist[1]).count('Print_BB_Run')

    # Now, create a category vector
    num_categories = 10
    category_vector = [0.0] * num_categories

    # Define category ranges
    control_flow_range = range(0, 11)
    arithmetic_range = range(11, 24)
    logic_range = range(24, 30)
    vector_operations_range = range(30, 33)
    memory_operations_range = range(35, 42)
    type_conversions_range = range(42, 55)
    comparison_operations_range = range(55, 57)
    data_flow_range = range(57, 60)
    function_call_range = range(60, 62)
    exception_handling_range = range(62, 65)

    # Update category vector based on feature vector
    for i, count in enumerate(features):
        if i in control_flow_range:
            category_vector[0] += count
        elif i in arithmetic_range:
            category_vector[1] += count
        elif i in logic_range:
            category_vector[2] += count
        elif i in vector_operations_range:
            category_vector[3] += count
        elif i in memory_operations_range:
            category_vector[4] += count
        elif i in type_conversions_range:
            category_vector[5] += count
        elif i in comparison_operations_range:
            category_vector[6] += count
        elif i in data_flow_range:
            category_vector[7] += count
        elif i in function_call_range:
            category_vector[8] += count
        elif i in exception_handling_range:
            category_vector[9] += count

    del features[64], features[63], features[61], features[59], features[54], features[38], features[23], \
        features[11], features[9], features[8], features[7], features[5], features[3]

    static_features_count = 0

    for i in range(len(features)):
        static_features_count += features[i]

    for i in range(len(features)):
        if features[i] != 0:
            features[i] = features[i] / static_features_count * 100

    dyn_features = [item * dyn for item in features]

    is_in_cycle = [0]
    if any(flist[0] in x for x in cycles_list):
        is_in_cycle[0] = 1

    is_in_simple_cycle = [0]
    if any(flist[0] in x for x in simple_cycles_list):
        is_in_simple_cycle[0] = 1

    is_bridges = [0]
    if any(flist[0] in x for x in bridges):
        is_bridges[0] = 1

    num_vars = [0, 0]
    num_vars[0] = str(flist[1]).count("%") - str(flist[1]).count("%\"")  # num local variables
    num_vars[0] = num_vars[0] / nn_vars

    num_vars[1] = str(flist[1]).count("@") - str(flist[1]).count("call") - str(flist[1]).count(
        'Print_BB_Run')  # num global variables

    features = features + dyn_features + is_in_cycle + num_vars + is_bridges

    BB_name_int = -1
    try:
        BB_name_int = int(BB_name)
    except:
        BB_name_int = -1

    category_vector.append(BB_name_int)
    cat_feat.append(BB_name_int)
    # print(cat_feat)
    return cat_feat

def to_feature_instruction(flist):
    helper = {"ret": 0, "br": 1, "switch": 2, "indirectbr": 3, "invoke": 4, "callbr": 5, "resume": 6, "catchswitch": 7,
              "catchret": 8, "cleanupret": 9, "unreachable": 10, "fneg": 11, "add": 12, "fadd": 13, "sub": 14,
              "fsub": 15, "mul": 16, "fmul": 17, "udiv": 18, "sdiv": 19, "fdiv": 20, "urem": 21, "srem": 22, "frem": 23,
              "shl": 24, "lshr": 25, "ashr": 26, "and": 27, "or": 28, "xor": 29, "extractelement": 30,
              "insertelement": 31, "shufflevector": 32, "extractvalue": 33, "insertvalue": 34, "alloca": 35, "load": 36,
              "store": 37, "fence": 38, "cmpxchg": 39, "atomicrmw": 40, "getelementptr": 41, "trunc": 42, "zext": 43,
              "sext": 44, "fptrunc": 45, "fpext": 46, "fptoui": 47, "fptosi": 48, "uitofp": 49, "sitofp": 50,
              "ptrtoint": 51, "inttoptr": 52, "bitcast": 53, "addrspacecast": 54, "icmp": 55, "fcmp": 56, "phi": 57,
              "select": 58, "freeze": 59, "call": 60, "va_arg": 61, "landingpad": 62, "catchpad": 63, "cleanuppad": 64}
    for key in helper:
        if key in flist:
            return helper[key]
    return -1

def to_feature(flist):
    """
    :param flist: get list of instructions that are called in this node of CFG(Basic block)
    then create a list of features that will be assigned to corresponding node of pyg graph node.
    :return:
    """
    str_list = str(flist[1])
    cat_feat = get_bb_sig(str_list)
    sub = "\""

    BB_name = str_list[str_list.find("{", 1) + 3:str_list.find(":", 2) - 2]
    dyn_run = get_dyn_run_of_bb(BB_name)
    # for i in range(100):
    #     cat_feat[i] = cat_feat[i]*dyn_run
    # # print(str_list)
    # # print(BB_name, "::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    # if BB_name:
    #     dyn = get_dyn_run_of_bb(int(BB_name))
    # else:
    #     dyn = 0
    #
    # features = [0] * 65
    # helper = {"ret": 0, "br": 1, "switch": 2, "indirectbr": 3, "invoke": 4, "callbr": 5, "resume": 6, "catchswitch": 7,
    #           "catchret": 8, "cleanupret": 9, "unreachable": 10, "fneg": 11, "add": 12, "fadd": 13, "sub": 14,
    #           "fsub": 15, "mul": 16, "fmul": 17, "udiv": 18, "sdiv": 19, "fdiv": 20, "urem": 21, "srem": 22, "frem": 23,
    #           "shl": 24, "lshr": 25, "ashr": 26, "and": 27, "or": 28, "xor": 29, "extractelement": 30,
    #           "insertelement": 31, "shufflevector": 32, "extractvalue": 33, "insertvalue": 34, "alloca": 35, "load": 36,
    #           "store": 37, "fence": 38, "cmpxchg": 39, "atomicrmw": 40, "getelementptr": 41, "trunc": 42, "zext": 43,
    #           "sext": 44, "fptrunc": 45, "fpext": 46, "fptoui": 47, "fptosi": 48, "uitofp": 49, "sitofp": 50,
    #           "ptrtoint": 51, "inttoptr": 52, "bitcast": 53, "addrspacecast": 54, "icmp": 55, "fcmp": 56, "phi": 57,
    #           "select": 58, "freeze": 59, "call": 60, "va_arg": 61, "landingpad": 62, "catchpad": 63, "cleanuppad": 64}
    #
    # for key in helper:
    #     features[helper[key]] += str(flist[1]).count(key)
    # features[60] = str(flist[1]).count('call') - str(flist[1]).count('Print_BB_Run')
    #
    # # Now, create a category vector
    # num_categories = 10
    # category_vector = [0.0] * num_categories
    #
    # # Define category ranges
    # control_flow_range = range(0, 11)
    # arithmetic_range = range(11, 24)
    # logic_range = range(24, 30)
    # vector_operations_range = range(30, 33)
    # memory_operations_range = range(35, 42)
    # type_conversions_range = range(42, 55)
    # comparison_operations_range = range(55, 57)
    # data_flow_range = range(57, 60)
    # function_call_range = range(60, 62)
    # exception_handling_range = range(62, 65)
    #
    # # Update category vector based on feature vector
    # for i, count in enumerate(features):
    #     if i in control_flow_range:
    #         category_vector[0] += count
    #     elif i in arithmetic_range:
    #         category_vector[1] += count
    #     elif i in logic_range:
    #         category_vector[2] += count
    #     elif i in vector_operations_range:
    #         category_vector[3] += count
    #     elif i in memory_operations_range:
    #         category_vector[4] += count
    #     elif i in type_conversions_range:
    #         category_vector[5] += count
    #     elif i in comparison_operations_range:
    #         category_vector[6] += count
    #     elif i in data_flow_range:
    #         category_vector[7] += count
    #     elif i in function_call_range:
    #         category_vector[8] += count
    #     elif i in exception_handling_range:
    #         category_vector[9] += count
    #
    # del features[64], features[63], features[61], features[59], features[54], features[38], features[23], \
    #     features[11], features[9], features[8], features[7], features[5], features[3]
    #
    # static_features_count = 0
    #
    # for i in range(len(features)):
    #     static_features_count += features[i]
    #
    # for i in range(len(features)):
    #     if features[i] != 0:
    #         features[i] = features[i] / static_features_count * 100
    #
    # dyn_features = [item * dyn for item in features]
    #
    # is_in_cycle = [0]
    # if any(flist[0] in x for x in cycles_list):
    #     is_in_cycle[0] = 1
    #
    # is_in_simple_cycle = [0]
    # if any(flist[0] in x for x in simple_cycles_list):
    #     is_in_simple_cycle[0] = 1
    #
    # is_bridges = [0]
    # if any(flist[0] in x for x in bridges):
    #     is_bridges[0] = 1
    #
    # num_vars = [0, 0]
    # num_vars[0] = str(flist[1]).count("%") - str(flist[1]).count("%\"")  # num local variables
    # num_vars[0] = num_vars[0] / nn_vars
    #
    # num_vars[1] = str(flist[1]).count("@") - str(flist[1]).count("call") - str(flist[1]).count(
    #     'Print_BB_Run')  # num global variables
    #
    # features = features + dyn_features + is_in_cycle + num_vars + is_bridges

    BB_name_int = -1
    try:
        BB_name_int = int(BB_name)
    except:
        BB_name_int = -1

    # category_vector.append(BB_name_int)
    cat_feat.append(BB_name_int)
    # print(cat_feat)
    return cat_feat


def print_pytorch_graph_info(input_pyg):
    print("~~~~====~~~~")
    print("graph: ", input_pyg)  # Data(edge_index=[2, 3], num_nodes=4)
    print("edge index: ", input_pyg.edge_index)
    print("num nodes: ", input_pyg.num_nodes)
    print("num features: ", input_pyg.num_features)
    graph = torch_geometric.utils.to_networkx(input_pyg)


def dynamic_bb_counter():
    """ update number of time a node of CFG has been called in dynamic execution of program
     it also update edge weight of program. this functions doesn't return anything. it just
     updates global variables edge_weight and dynamic_bb_run"""
    global edge_weight
    global dynamic_bb_run
    f2 = open(raw_location + FolderName + "/llfi/baseline/bb_trace.prof.txt", "r")
    last_line = "-1"
    for line in f2:
        # if line != last_line:
            edge_weight[str(last_line).strip() + "->" + str(line).strip()] = \
                edge_weight.get(str(last_line).strip() + "->" + str(line).strip(), 0) + 1
            last_line = line
            dynamic_bb_run[int(line)] = dynamic_bb_run.get(int(line), 0) + 1


# from torch_geometric.data import Data, HeteroData
# from torch_geometric.data.datapipes import functional_transform
# from torch_geometric.transforms import remove_isolated_nodes as pxe
import torch_geometric.transforms as T
from torch_geometric.utils import remove_isolated_nodes as rin

# transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])


def make_no_runned_nodes_isolate(pyg_dg):
    edges = pyg_dg.edge_index
    list1 = []
    list2 = []
    for i in range(len(edges[0])):
        item1 = pyg_dg.x[edges[0][i]]
        item2 = pyg_dg.x[edges[1][i]]
        a1 = sum(item1[0:65])
        a2 = sum(item2[0:65])
        b1 = sum(item1[65:130])
        b2 = sum(item2[65:130])
        c1 = sum(item1)
        c2 = sum(item2)
        if (c1 != 0 and b1 == 0) or (c2 != 0 and b2 == 0):
            pass
        else:
            list1.append(edges[0][i])
            list2.append(edges[1][i])

    dt = torch.tensor([list1, list2])
    pyg_dg.edge_index = dt
    return pyg_dg


import statistics


def get_node_stats(nums):
    """
    ???
    :param nums:
    :return:
    """
    feature = []
    feature.append(max(nums))
    # feature.append(1)
    feature.append(min(nums))
    feature.append(round(sum(nums) / len(nums), 2))
    feature.append(statistics.median(nums))
    if 1 < len(nums):
        feature.append(round(statistics.variance(nums), 2))
    else:
        feature.append(round(0, 2))

    feature.append(statistics.mode(nums))
    return feature


def get_edges_statistics(edges):
    """
    :param edges: list of edges of graph
    :return: some statistics of edges based
    on input and output degree of edges
    """
    in_to_plot = []
    out_to_plot = []
    output_degree = edges[0].tolist()
    out_count = {}
    input_degree = edges[1].tolist()
    in_count = {}
    for element in output_degree:
        if element in out_count:
            out_count[element] += 1
        else:
            out_count[element] = 1

    for element in input_degree:
        if element in in_count:
            in_count[element] += 1
        else:
            in_count[element] = 1
    in_to_plot.append(list(in_count.values()))

    out_to_plot.append(list(out_count.values()))
    if len(in_to_plot[0]) != 0 and len(out_to_plot[0]) != 0:
        feature = [max(in_to_plot[0]),
                   min(in_to_plot[0]),
                   round(sum(in_to_plot[0]) / len(in_to_plot[0]), 2),
                   statistics.median(in_to_plot[0]),
                   round(statistics.variance(in_to_plot[0]), 2),
                   statistics.mode(in_to_plot[0]),
                   max(out_to_plot[0]),
                   min(out_to_plot[0]),
                   round(sum(out_to_plot[0]) / len(out_to_plot[0]), 2),
                   statistics.median(out_to_plot[0]),
                   round(statistics.variance(out_to_plot[0]), 2),
                   statistics.mode(out_to_plot[0])]
    else:
        feature = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    return feature


def coo_to_adjacency_list(row, col):
    """
    Convert COO format to an adjacency list.
    """
    row = row.tolist()
    col = col.tolist()
    m1 = max(row)
    m2 = max(col)
    m = max(m1, m2)
    r = [0] * (m + 1)
    graph = [r] * (m + 1)
    for i in range(len(row)):
        graph[row[i]][col[i]] = 1

    return graph


def dfs_paths(graph, start, end, path=[], visited=set()):
    """
    Depth-First Search to find all paths between start and end nodes.

    Args:
    - graph: The adjacency matrix representing the graph.
    - start: The starting node.
    - end: The destination node.
    - path: The current path being traversed.
    - visited: Set of visited nodes to prevent cycles.

    Returns:
    - List of all paths between start and end nodes.
    """
    path = path + [start]
    visited.add(start)
    if start == end:
        return [path]
    if not graph[start]:
        return []
    paths = []
    for node, connected in enumerate(graph[start]):
        if connected and node not in visited:
            new_paths = dfs_paths(graph, node, end, path, visited)
            for new_path in new_paths:
                paths.append(new_path)
    visited.remove(start)
    return paths


def digraph_to_pytorch_tensor_old(input_graph):
    # make simple graph
    new_dg = nx.DiGraph()
    new_dg.add_nodes_from(input_graph.nodes)
    new_dg.add_edges_from(input_graph.edges)

    # find some graph features
    global cycles_list
    cycles_list = nx.cycle_basis(new_dg.to_undirected())
    global simple_cycles_list
    # simple_cycles_list = sorted(nx.simple_cycles(new_dg))
    global bridges
    bridges = list(nx.bridges(new_dg.to_undirected()))
    global nn_vars
    nn_vars = 1  # number of varibales in whole program

    for e in list(input_graph.nodes.data("label")):
        nn_vars = nn_vars + str(e[1]).count("%") - str(e[1]).count("%\"")
    Res2 = [input_graph.degree[ele] for ele in list(input_graph.nodes)]  # Res2 is every nodes degree
    pyg_dg = from_networkx(new_dg)
    global edge_weight
    edge_weight = {}
    dynamic_bb_counter()
    Res = [to_feature(ele) for ele in list(input_graph.nodes.data("label"))]
    pyg_dg.x = torch.tensor(Res)
    if pyg_dg.num_nodes > 2:
        pyg_dg = transform(pyg_dg)
    feature = get_edges_statistics(pyg_dg.edge_index)  # 12
    nm_edges = pyg_dg.num_edges
    graph_level_features = []
    graph_level_features = graph_level_features + [len(bridges) / (nm_edges + 1)]
    # graph_level_features = graph_level_features + [nn_vars]
    graph_level_features = graph_level_features + get_node_stats(Res2)  # 6
    graph_level_features = graph_level_features + feature  # 12
    graph_level_features = graph_level_features + [len(simple_cycles_list) / (nm_edges + 1)]
    graph_level_features = graph_level_features + [len(cycles_list) / (nm_edges + 1)]

    logging.info("feature: " + str(feature))
    logging.info(str(pyg_dg.x.shape))
    logging.info(str(type(pyg_dg.x)))
    pyg_dg.graph_features = torch.tensor(graph_level_features, dtype=torch.float)
    logging.info(str(pyg_dg.x.shape))
    pinFI_result = open("../../data/FI_result/FI_result_3000", "r")
    lines = pinFI_result.readlines()

    for line in lines:
        print(line)
        if line.find("  " + filename) != -1:
            z = torch.tensor([float(edx) for edx in (line.split("=")[1]).split(",")], dtype=torch.float)
            pyg_dg.y = z
            # 0 SDC, 1 MASK, 2 Crash, 4 all three
            pyg_dg.sdc = z[0]
            pyg_dg.mask = z[1]
            pyg_dg.crash = z[2]

    pyg_dg.name = filename
    w_edge = []
    for en in range(pyg_dg.num_edges):
        a = str(int(pyg_dg.x[pyg_dg.edge_index[0][en]][-1])).strip()
        b = str(int(pyg_dg.x[pyg_dg.edge_index[1][en]][-1])).strip()
        # print((a + "->" + b))
        w_edge.append(edge_weight.get(a + "->" + b, 0))

    pyg_dg.edge_weight = torch.tensor(w_edge, dtype=torch.float)

    # add edge type
    edge_type = []  #0=control_flow , 1= data_flow
    for edge in input_graph.edges(data=True):
        # edge is a tuple (u, v, data)
        source_node = edge[0]
        target_node = edge[1]
        edge_data = edge[2]

        # Access the type attribute of the edge, if it exists
        if 'type' in edge_data:
            edge_type.append(1)
            # edge_type = edge_data['type']
            # print(f"Edge from {source_node} to {target_node} has type {edge_type}")
        else:
            edge_type.append(0)

    pyg_dg.edge_type = torch.tensor(edge_type, dtype=torch.float)

    logging.info("pyg_dg: " + str(pyg_dg))
    logging.info("**==" * 10)
    graph_adj = coo_to_adjacency_list(pyg_dg.edge_index[0], pyg_dg.edge_index[1])
    logging.info(str(graph_adj))
    for i in range(len(pyg_dg.edge_type)):
        if pyg_dg.edge_type[i] == 1 and pyg_dg.edge_weight[i] != 0:
            pyg_dg.edge_type[i] = 1  #redundant orginal 2
        elif pyg_dg.edge_type[i] == 1 and pyg_dg.edge_weight[i] == 0:
            pyg_dg.edge_weight[i] = 0  #redundant orginal 1

    return pyg_dg


def get_FI_result(filename):
    pinFI_result = open("../../data/FI_result/" + FI_result_location, "r")
    lines = pinFI_result.readlines()
    for line in lines:
        if line.find("  " + filename) != -1:
            z = torch.tensor([float(edx) for edx in (line.split("=")[1]).split(",")], dtype=torch.float)
            return z
            # 0 SDC, 1 MASK, 2 Crash, 4 all three


def digraph_to_pytorch_tensor(input_graph):
    """
    we convert directed networkx graph to PyTorch geometric
    graph representation.
    :param input_graph: directed networkx graph
    :return: pytorch geometric graph
    """
    global edge_weight
    edge_weight = {}
    # make simple graph
    new_dg = nx.DiGraph()
    new_dg.add_nodes_from(input_graph.nodes)
    new_dg.add_edges_from(input_graph.edges(data=True))
    dynamic_bb_counter()  #this function update edge weight and node count

    default_attributes = {'type': 'control_flow'}
    # Add default attributes to edges that are missing them
    for u, v, data in new_dg.edges(data=True):
        for attr, default_value in default_attributes.items():
            if attr not in data:
                data[attr] = default_value

    values = {}
    for n in input_graph.nodes.data("label"):
        values[n[0]] = to_feature(n)

    nx.set_node_attributes(new_dg, values, name="x")

    pyg_dg = from_networkx(new_dg)

    # Res = [to_feature(ele) for ele in list(input_graph.nodes.data("label"))]
    # pyg_dg.x = torch.tensor(Res)
    if pyg_dg.num_nodes > 2:
        try:
            pyg_dg = transform(pyg_dg)
        except:
            print("error")

    pinFI_result = get_FI_result(filename)
    pyg_dg.y = pinFI_result
    # if isinstance(pinFI_result, list):
    pyg_dg.sdc = pinFI_result[0]  #SDC
    pyg_dg.mask = pinFI_result[1]  #Mask
    pyg_dg.crash = pinFI_result[2]  #Interruption
    pyg_dg.name = filename

    w_edge = []
    for en in range(pyg_dg.num_edges):
        a = str(int(pyg_dg.x[pyg_dg.edge_index[0][en]][-1])).strip()
        b = str(int(pyg_dg.x[pyg_dg.edge_index[1][en]][-1])).strip()
        if pyg_dg.type[en] == "control_flow":
            w_edge.append(edge_weight.get(a + "->" + b, 0) + 0)
        else:
            w_edge.append(1)

    pyg_dg.edge_weight = torch.tensor(w_edge, dtype=torch.float)
    delattr(pyg_dg, "type")
    return pyg_dg

def get_instruction_node_text(bb, llindex):
    with open(raw_location + FolderName + "/llfi/" + FolderName + "-llfi_index.ll", 'r') as f:
        for line in f:
            # print("-")
            line = line.strip()
            o = line.split("!llfi_index !")
            if len(o)>1:
                if o[1] == llindex:
                    return line
    return "empty"

def get_inst_FI_res(ll_index):
    # print(ll_index)
    with open("../../data/PER_INST_FI/result/"+FI_result_location+"/" + FolderName , 'r') as f:
        for line in f:
            # print(int(line.split("=")[0].strip()))
            if "NoFile"!=line.split("=")[0].strip():
                if int(line.split("=")[0].strip()) == int(ll_index):
                    # print(line.split("=")[1].split(";")[0].split(","),"<<<<<<<<<<<<<<<<<<<")
                    z = [float(edx) for edx in (line.split("=")[1].split(";")[0]).split(",")]
                    # print(z,"???????????????????")
                    return z, 1
    with open("../../data/raw/"+FI_result_location+"/" + FolderName +"/llfi.log.compilation.txt" , 'r') as f:
        for line in f:
            if " !llfi_index !"+ll_index in line:
                return [0,0,0], 0 #"Error FI"
    return "No FI", 0
    # return [0.0, 1.0, 0.0]


def digraph_to_pytorch_tensor_instruction(input_graph):
    """
    we convert directed networkx graph to PyTorch geometric
    graph representation.
    :param input_graph: directed networkx graph
    :return: pytorch geometric graph
    """
    print(input_graph, "...")
    global edge_weight
    edge_weight = {}
    # make simple graph
    new_dg = nx.DiGraph()
    new_dg.add_nodes_from(input_graph.nodes)
    new_dg.add_edges_from(input_graph.edges(data=True))
    dynamic_bb_counter()  #this function update edge weight and node count

    default_attributes = {'type': 'control_flow'}
    # Add default attributes to edges that are missing them
    for u, v, data in new_dg.edges(data=True):
        for attr, default_value in default_attributes.items():
            if attr not in data:
                data[attr] = default_value

    values = {}
    FIs = {}
    nodessdc = {}
    cnt = 0
    ttl = 0
    status = {}
    for n in input_graph.nodes.data("label"):
        # try:
        # print(n[0])
        try:
            node_inst = get_instruction_node_text(str(n[0]).split("_")[0], str(n[0]).split("_")[2])
            FI_res, stts = get_inst_FI_res(str(n[0]).split("_")[2])
            # print(node_inst, "<<<<")
        except:
            node_inst="unreachable, !llfi_index !-1 "
            FI_res, stts = [0,1,0],0
        # except:
        #     node_inst = 'empty'
        #     FI_res = [0.0, 1.0, 0.0]
        ttl += 1
        if "FI" in FI_res:
            cnt += 1
        try:
            values[n[0]] = [to_feature_instruction(node_inst)]+[int(str(n[0]).split("_")[0])]
        except:
            values[n[0]] = [to_feature_instruction(node_inst)] + [-1]
        FIs[n[0]] = FI_res
        status[n[0]] = stts
        ns = 0
        if isinstance(FI_res[0], float) and FI_res[0] > 0:
            ns = 1
        nodessdc[n[0]] = ns
        # print("dynamic_bb_run: ", dynamic_bb_run.keys())
        # print("n[0]: ", n[0])
        # print("int(str(n[0]).split()[0]): ", int(str(n[0]).split("_")[0]))
        if int(str(n[0]).split("_")[0]) in dynamic_bb_run.keys() and FI_res == "No FI":
            FI_res = [0,1,0]
            FIs[n[0]] = [0,1,0] # Error FI IIIIIIIIIIIII
        if int(str(n[0]).split("_")[0]) not in dynamic_bb_run.keys():
            FI_res = "No FI"
            FIs[n[0]] = "No FI"
        # print(str(n[0]), ":", FI_res, ":", values[n[0]])
    # print(cnt,",", ttl, ":::", (ttl-cnt)/ttl)
    # print(values)
    # print(FIs)

    nx.set_node_attributes(new_dg, values, name="x")
    nx.set_node_attributes(new_dg, FIs, name="FIs")
    nx.set_node_attributes(new_dg, status, name="status")
    nx.set_node_attributes(new_dg, nodessdc, name="nodessdc")

    # Show 1
    pyg_dg = from_networkx(new_dg)
    # print("1:", pyg_dg)
    # node_colors = ["red" if new_dg.nodes[node]["FIs"] == "No FI" else "green" for node in new_dg]
    # plt.figure(figsize=(10, 8))
    # nx.draw(new_dg, with_labels=True, node_color=node_colors, edge_color="gray", node_size=700, font_size=10)
    # plt.title("Graph Visualization with FIs Attribute Highlighted")
    # plt.show()

    # Filter out nodes with FIs attribute set to "NoFI"
    nodes_to_remove = [node for node, attr in new_dg.nodes(data=True) if attr["FIs"] == "No FI"]
    new_dg.remove_nodes_from(nodes_to_remove)

    # Keep only the largest connected component
    largest_cc = max(nx.weakly_connected_components(new_dg), key=len)
    new_dg = new_dg.subgraph(largest_cc).copy()

    pyg_dg = from_networkx(new_dg)
    # Show 2
    # print("2:", pyg_dg)
    # node_colors = ["blue" if new_dg.nodes[node]["FIs"] == "Error FI" else "green" for node in new_dg]
    # plt.figure(figsize=(10, 8))
    # nx.draw(new_dg, with_labels=True, node_color=node_colors, edge_color="gray", node_size=700, font_size=10)
    # plt.title("Graph Visualization with FIs Attribute Highlighted")
    # plt.show()

    # Res = [to_feature(ele) for ele in list(input_graph.nodes.data("label"))]
    # pyg_dg.x = torch.tensor(Res)
    # if pyg_dg.num_nodes > 2:
    #     try:
    #         pyg_dg = transform(pyg_dg)
    #     except:
    #         print("error")

    pinFI_result = get_FI_result(filename)
    pyg_dg.y = pinFI_result
    # if isinstance(pinFI_result, list):
    pyg_dg.sdc = pinFI_result[0]  #SDC
    pyg_dg.mask = pinFI_result[1]  #Mask
    pyg_dg.crash = pinFI_result[2]  #Interruption
    pyg_dg.name = filename

    w_edge = []
    # print("edge_weight:", edge_weight)
    for en in range(pyg_dg.num_edges):
        a = str(int(pyg_dg.x[pyg_dg.edge_index[0][en]][-1])).strip()
        b = str(int(pyg_dg.x[pyg_dg.edge_index[1][en]][-1])).strip()
        # print(a,"->",b)
        if pyg_dg.type[en] == "control_flow":
            w_edge.append(edge_weight.get(a + "->" + b, 0) + 0)
            # print(edge_weight.get(a + "->" + b, 0) + 0)
        else:
            w_edge.append(1)
            # print("s")
    # return input_graph
    pyg_dg.edge_weight = torch.tensor(w_edge, dtype=torch.float)
    print(pyg_dg)
    delattr(pyg_dg, "type")
    return pyg_dg

def remove_no_run_nodes(input_graph):
    G = torch_geometric.utils.to_networkx(input_graph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G, 'state')
    edge_labels = nx.get_edge_attributes(G, 'state')
    plt.show()


def get_inst_set_of_node_used_by_find_data_dep(cfg, node):
    """
    in finding data dependency process we have a graph
    we call it cfg here. each node represent a basic block,
    so it consist from set of instructions. we use this function
    to get list of instructions of node in cfg
    :param cfg: input graph
    :param node: our specific node
    :return: instructions of the node
    """
    for t in list(cfg.nodes.data("label")):
        if t[0] == str(node):
            return str(t[1])
    return ""


def get_all_successors(graph, node):
    """
    we need successors of a node in graph, but it's time-consuming
    to get them using networkx, so we get name of this node for example 20
    and if there is any node with higher name like 30,50,60 we say they
    are successors of current node
    :param graph:
    :param node:
    :return:
    """
    node_name = get_inst_set_of_node_used_by_find_data_dep(graph, node).split("\l")[0].strip("\"{:\\ ")
    all_successors = list(graph.nodes())
    send = []
    for a in all_successors:
        name_a = get_inst_set_of_node_used_by_find_data_dep(graph, a).split("\l")[0].strip("\"{:\\ ")
        if int(name_a) > int(node_name):
            send.append(a)
    return send


def get_used_variable_set_of_node(cfg, node, just_read_inst=False):
    """
    instructions that are used in the node called node in graph called cfg
    will be returned
    :param cfg:
    :param node:
    :param just_read_inst:
    :return:
    """
    instructions = get_inst_set_of_node_used_by_find_data_dep(cfg, node).split("\l")
    defined_variables = set()
    # Iterate over the instructions to find defined variables
    for instruction in instructions:
        if just_read_inst:
            if any(op in instruction for op in
                   ['load', 'icmp', 'fcmp', 'select', 'extractelement', 'getelementptr', 'va_arg', 'icmp', 'fcmp',
                    'call']):
                parts = instruction.split("=")
                matches = re.findall(r'%\w+', instruction)
                for m in matches:
                    defined_variables.add(m)
        else:
            parts = instruction.split("=")
            matches = re.findall(r'%\w+', instruction)
            for m in matches:
                defined_variables.add(m)

    return defined_variables


def get_result_variable_set_of_node(cfg, node, just_read_inst=False):
    """ this function returns variable that defined in the node (basic block)
    based on pattern %x = s.th.
    :param cfg: input graph
    :param node: selected node
    :param just_read_inst: check for group of instruction that are just read
    :return: set of instructions that are defined in this node(BB)
    """
    instructions = get_inst_set_of_node_used_by_find_data_dep(cfg, node).split("\l")
    defined_variables = set()
    # Iterate over the instructions to find defined variables
    for instruction in instructions:
        if just_read_inst:
            if any(op in instruction for op in
                   ['load', 'icmp', 'fcmp', 'select', 'extractelement', 'getelementptr', 'va_arg', 'icmp', 'fcmp',
                    'call']):
                # Split the instruction to get the operation and operands
                parts = instruction.split("=")
                if len(parts) > 1 and "%" in parts[0]:
                    defined_variable = parts[0].strip()
                    defined_variables.add(defined_variable)
        else:
            parts = instruction.split("=")
            if len(parts) > 1 and "%" in parts[0]:
                defined_variable = parts[0].strip()
                defined_variables.add(defined_variable)

    return defined_variables


def find_data_dependencies(cfg):
    """
    to make our graphs more helpful we add data dependency edges
    to graphs
    :param cfg: is out input graph. because it only includes control flow edges
    we call it cfg
    :return: new graph_export_location with control and data flow edges
    """
    logging.info("Hi this is find_data_dependencies")
    data_dependencies = []
    # return cfg
    for node in cfg.nodes():
        logging.info("basic block name:" +
                     get_inst_set_of_node_used_by_find_data_dep(cfg, node).split("\l")[0].strip("\"{:\\ "))
        defined_variables = get_result_variable_set_of_node(cfg, node)
        logging.info("defined_variables of this:" + str(defined_variables))

        all_successors = set()
        all_successors = get_all_successors(cfg, node)
        logging.info("successors of current node:")

        for successors in all_successors:
            successors_instructions = get_used_variable_set_of_node(cfg, successors, True)
            logging.info("successors_name:" +
                         get_inst_set_of_node_used_by_find_data_dep(cfg, successors).split("\l")[0].strip("\"{:\\ "))
            logging.info("successors_instructions:" + str(successors_instructions))
            if successors_instructions & defined_variables:
                logging.info("Yes")
                data_dependencies.append((node, successors))
                cfg.add_edge(node, successors, type="data_flow")

        logging.info(data_dependencies)
        logging.info("end node")
    return cfg


# 1A_-_Theater_Square-llfi_index.ll
# parse ll index file
def parse_ll_index_file():
    bb_to_instructions = {}
    llfi_index_pattern = re.compile(r'!llfi_index !(\d+)')  # Regex to capture the llfi_index number

    with open(raw_location + FolderName + "/llfi/" + FolderName + "-llfi_index.ll", 'r') as f:
        current_bb = None
        for line in f:
            line = line.strip()
            # Identify the basic block header
            if re.match(r'^"\d+":', line):
                current_bb = re.match(r'^"(\d+)":', line).group(1)
                bb_to_instructions[current_bb] = []
            # Collect instructions under the current basic block
            # elif current_bb and (line.startswith('%') or line.startswith('call') or 'ret' in line):
            elif current_bb:
                # print("current_bb", current_bb)
                # Try to find llfi_index in the line
                llfi_match = llfi_index_pattern.search(line)
                # print(llfi_match)
                if llfi_match:
                    llfi_index = llfi_match.group(1)  # Extract the number after !llfi_index
                    # Append llfi_index to the instruction
                    # line_with_index = f"{line} (llfi_index {llfi_index})"
                    # print(llfi_index)
                    bb_to_instructions[current_bb].append(int(llfi_index))

                else:
                    # If no llfi_index, just keep the instruction as-is
                    llfi_index = 99

                # bb_to_instructions[current_bb].append(int(llfi_index))
    return bb_to_instructions


def extract_node_name(G, node):
    # Match the node name inside the first set of double quotes
    for i in G.nodes.data("label"):
        if i[0] == node:
            node_text =  i[1]
    match = node_text.split("\l")[0].strip("\"{:\\ ")
    if match:
        return match
    return None  # Return None if no match is found

def get_min_max_bb(bb_to_instructions, node_name):
    try:
        instructions = bb_to_instructions[node_name]
    except:
        print("Erroooooooooooooooooor")
        instructions = bb_to_instructions[str(int(node_name)-10)]
    return min(instructions), max(instructions)

def replace_bb_with_instructions(graph, bb_to_instructions):
    new_graph = nx.DiGraph()

    for node in list(graph.nodes()):
        node_name = extract_node_name(graph, node)  # Extract the actual node name

        if node_name in bb_to_instructions:
            instructions = bb_to_instructions[node_name]

            if 0 in instructions:
                instructions.remove(0)

            # Create instruction nodes
            prev_instruction = None
            for idx, instruction in enumerate(instructions):
                instruction_node = f'{node_name}_inst_{instruction}'
                new_graph.add_node(instruction_node, label=instruction)

                if prev_instruction:
                    new_graph.add_edge(prev_instruction, instruction_node)

                prev_instruction = instruction_node

            # Maintain connections between basic blocks and the first/last instruction nodes
            for succ in list(graph.successors(node)):
                succ_name = extract_node_name(graph, succ)
                succ_name = str(succ_name) + "_inst_" + str(get_min_max_bb(bb_to_instructions, succ_name)[0])

                # Check if the edge has type "data_flow"
                edge_type = graph.get_edge_data(node, succ).get('type', None)
                if edge_type == "data_flow":
                    new_graph.add_edge(prev_instruction, succ_name, type="data_flow")
                else:
                    new_graph.add_edge(prev_instruction, succ_name)

            for pred in list(graph.predecessors(node)):
                pred_name = extract_node_name(graph, pred)
                pred_name = str(node_name) + "_inst_" + str(get_min_max_bb(bb_to_instructions, pred_name)[1])
                node_name_name = str(node_name) + "_inst_" + str(get_min_max_bb(bb_to_instructions, node_name)[0])

                # Check if the edge has type "data_flow"
                edge_type = graph.get_edge_data(pred, node).get('type', None)
                if edge_type == "data_flow":
                    new_graph.add_edge(pred_name, node_name_name, type="data_flow")
                else:
                    new_graph.add_edge(pred_name, node_name_name)

        else:
            print("node is not in list:", graph.nodes[node]['label'])
            new_graph.add_node(node, label=graph.nodes[node]['label'])
            for succ in list(graph.successors(node)):
                # Check if the edge has type "data_flow"
                edge_type = graph.get_edge_data(node, succ).get('type', None)
                if edge_type == "data_flow":
                    new_graph.add_edge(node, succ, type="data_flow")
                else:
                    new_graph.add_edge(node, succ)

    return new_graph


def replace_bb_with_instructions0(graph, bb_to_instructions):
    new_graph = nx.DiGraph()

    for node in list(graph.nodes()):
        node_name = extract_node_name(graph, node)  # Extract the actual node name
        # print(node_name)
        if node_name in bb_to_instructions:
            instructions = bb_to_instructions[node_name]
            # print(instructions)
            if 0 in instructions:
                instructions.remove(0)
            # Create instruction nodes
            prev_instruction = None
            for idx, instruction in enumerate(instructions):
                # print("instruction:", instruction)
                instruction_node = f'{node_name}_inst_{instruction}'
                new_graph.add_node(instruction_node, label=instruction)

                if prev_instruction:
                    new_graph.add_edge(prev_instruction, instruction_node)

                prev_instruction = instruction_node

            # Maintain connections between basic blocks and the first/last instruction nodes
            # prv_name = extract_node_name(graph, prev_instruction)
            # prv_name = prv_name+"_inst_"+get_min_max_bb(bb_to_instructions, prv_name)[1]


            for succ in list(graph.successors(node)):
                succ_name = extract_node_name(graph, succ)
                # print(succ, "<=>", succ_name)
                succ_name = str(succ_name) + "_inst_" + str(get_min_max_bb(bb_to_instructions, succ_name)[0])
                # new_graph.add_edge(prev_instruction, succ)
                new_graph.add_edge(prev_instruction, succ_name)
            for pred in list(graph.predecessors(node)):
                pred_name = extract_node_name(graph, pred)
                pred_name = str(node_name) + "_inst_" + str(get_min_max_bb(bb_to_instructions, pred_name)[1])
                # node_name_name = extract_node_name(graph, node_name)
                node_name_name = str(node_name) + "_inst_" + str(get_min_max_bb(bb_to_instructions, node_name)[0])
                # new_graph.add_edge(pred, f'{node_name}_inst_0')
                new_graph.add_edge(pred_name, node_name_name)
        else:
            print("node is not in list:", graph.nodes[node]['label'])
            new_graph.add_node(node, label=graph.nodes[node]['label'])
            for succ in list(graph.successors(node)):
                new_graph.add_edge(node, succ)

    return new_graph



def main(Folder_name, rawLocation, export_location, FI_result_loc, nameindex):
    """ try to convert raw data to graphs that are useful for pytorch geometric """
    # we set global variables because other functions also need these variables
    global FolderName, raw_location, graph_export_location, FI_result_location
    FolderName = Folder_name
    raw_location = rawLocation
    graph_export_location = export_location
    FI_result_location = FI_result_loc
    print("==" * 10)
    print("try convert raw data to graph: ", FolderName)
    dot_files = get_dot_files_name_list()
    func_names = convert_dot_names_to_function_names(dot_files.copy())
    graphs = get_graphs_dict(dot_files)
    func_names.remove("main")
    logging.info("func_names: " + str(func_names))
    DG_main = nx.DiGraph(
        nx.nx_pydot.read_dot(raw_location + FolderName + "/dotfiles/cfg.main.dot"))  # Read Graph

    logging.info(DG_main.nodes.data("label"))
    DG_main = remove_ports(DG_main)
    DG_main = find_data_dependencies(DG_main)

    for i in graphs:
        logging.info("Graph [" + i + "]")
        logging.info("unmodified")
        logging.info("Number of nodes:" + str(graphs[i].number_of_nodes()))
        logging.info("Number of edges:" + str(graphs[i].number_of_edges()))
        graphs[i] = remove_ports(graphs[i])
        logging.info("remove_ports")
        logging.info("Number of nodes:" + str(graphs[i].number_of_nodes()))
        logging.info("Number of edges:" + str(graphs[i].number_of_edges()))
        graphs[i] = find_data_dependencies(graphs[i])
        logging.info("find_data_dependencies")
        # print(graphs[i].edges.data("label"))
        logging.info("Number of nodes:" + str(graphs[i].number_of_nodes()))
        logging.info("Number of edges:" + str(graphs[i].number_of_edges()))

    node_gen = 0
    will_continue = True  # we want to visit all nodes
    visited_nodes = []  # we didn't change any nodes yet
    num_iterations = 0
    while will_continue:
        num_iterations = num_iterations + 1
        will_continue = False  # we revisit nodes if any changes happen in this iteration /
        # if any changes happened we will
        s = copy.deepcopy(graphs['main'].nodes.data("label"))
        for i in s:
            if i[0] not in visited_nodes:
                list_of_called_functions_in_this_node = check_existence(i[1], func_names)
                for called_func in list_of_called_functions_in_this_node:
                    visited_nodes.append(
                        i[0])  # we will make changes to this node, so we don't want to visit this again
                    will_continue = True  # we will make changes to graph, so we should travers it again
                    DG_main = dg_duplicate_node(DG_main, i[0], 'new' + str(node_gen))  # we duplicate current node
                    DG_main = add_graph_to_node(DG_main, graphs[called_func], i[0], 'new' + str(node_gen))
                    visited_nodes.append('new' + str(node_gen))
                    node_gen = node_gen + 1
        graphs['main'] = DG_main

    logging.info("after merging")
    logging.info("Number of nodes:" + str(DG_main.number_of_nodes()))
    logging.info("Number of edges:" + str(DG_main.number_of_edges()))
    bb_to_instructions = parse_ll_index_file()
    print(DG_main)

    # print(DG_main.edges)
    # print(DG_main.edges.data("type"))
    # print(DG_main.nodes.data("label"))
    # for nds in DG_main:
    #     print(nds)
    #     print(extract_node_name(DG_main, nds))
    DG_main = replace_bb_with_instructions(DG_main, bb_to_instructions)
    # print(DG_main)
    # print(DG_main.edges.data("type"))
    # print(DG_main.nodes.data("label"))
    # print(DG_main.edges)
    # for node in list(DG_main.nodes()):
    #     print(node)

    pyg = digraph_to_pytorch_tensor_instruction(DG_main)
    pyg.name = pyg.name + nameindex
    # print(pyg.FIs[0:100])
    print(sum(pyg.nodessdc),"____of____", len(pyg.nodessdc))
    if 'y' in pyg:
        torch.save(pyg, graph_export_location + FolderName + nameindex)


if __name__ == '__main__':
    print("Start Of Program:")
    logging.info("This is an info message")
    # # # ---------------------------------
    # print("convert raw train data to codeforceS1 graphs")
    # directory = "../../data/raw/codeforceS1/"
    # for filename in os.listdir(directory):
    #     main(filename, "../../data/raw/codeforceS1/", "../data_i/graphs/train/", "codeforceS1", "")
    # # # # # ---------------------------------
    # print("convert raw train data to codeforceS2 graphs")
    # directory = "../../data/raw/codeforceS2/"
    # for filename in os.listdir(directory):
    #     main(filename, "../../data/raw/codeforceS2/", "../data_i/graphs/train/", "codeforceS2" , "2")
    # # # # # ---------------------------------
    # print("convert raw train data to progs100 graphs")
    # directory = "../../data/raw/progs100/"
    # for filename in os.listdir(directory):
    #     main(filename, "../../data/raw/progs100/", "../data_i/graphs/train/", "progs100" , "")
    # # # #
    # # #
    # # # # ---------------------------------
    print("convert raw test data to test graphs")
    directory = "../../data/raw/test/"
    for filename in os.listdir(directory):
        start = timeit.default_timer()
        main(filename, "../../data/raw/test/", "../data_i/graphs/test/", "test", "")
        stop = timeit.default_timer()
        print('Time: ', stop - start)





    # # ---------------------------------
    # print("convert raw test data to multi blackscholes")
    # directory = "../data/raw/multi_blackscholes/"
    # for filename in os.listdir(directory):
    #     start = timeit.default_timer()
    #     main(filename, "../data/raw/multi_blackscholes/", "../data/graphs/multi_blackscholes/", "multi_blackscholes", "")
    #     stop = timeit.default_timer()
    #     print('Time: ', stop - start)
    # # ---------------------------------
    # print("convert raw test data to multi hotspot")
    # directory = "../data/raw/multi_hotspot/"
    # for filename in os.listdir(directory):
    #     start = timeit.default_timer()
    #     main(filename, "../data/raw/multi_hotspot/", "../data/graphs/multi_hotspot/", "multi_hotspot", "")
    #     stop = timeit.default_timer()
    #     print('Time: ', stop - start)
    # # ---------------------------------
    # print("convert raw test data to multi bfs")
    # directory = "../data/raw/multi_bfs/"
    # for filename in os.listdir(directory):
    #     start = timeit.default_timer()
    #     main(filename, "../data/raw/multi_bfs/", "../data/graphs/multi_bfs/", "multi_bfs", "")
    #     stop = timeit.default_timer()
    #     print('Time: ', stop - start)
