
import networkx as nx
import scipy as sp
import numpy as np
import math
import tqdm
from timeit import default_timer



def maximum_independent_set(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    for v in G.nodes:
        copy_G_without_v = G.copy()
        copy_G_without_v.remove_node(v)
        copy_G_without_Nv = G.copy()
        neighbors_v = list(G.neighbors(v))
        copy_G_without_Nv.remove_nodes_from(neighbors_v)
        copy_G_without_Nv.remove_node(v)
        return max(maximum_independent_set(copy_G_without_v), maximum_independent_set(copy_G_without_Nv) + 1)


def approx_independent_set(G:nx.Graph) -> int:
        independent_set = set()

        for node in sorted(G.degree(), key=lambda x: x[1], reverse=False):
            neighbors = G.neighbors(node[0])
            if not any(neighbor in independent_set for neighbor in neighbors):
                independent_set.add(node[0])

        return len(independent_set)



def maximum_degree(graph) -> int:
    max_degree = 0
    for node in graph.nodes():
        degree = graph.degree[node]
        if degree > max_degree:
            max_degree = degree

    return max_degree


def minimum_degree(graph) -> int:
    min_degree = float('inf')

    for node in graph.nodes():
        degree = graph.degree[node]
        if degree < min_degree:
            min_degree = degree

    return min_degree


def Kwok_bound(graph: nx.Graph) -> float:
    return float(np.around(graph.number_of_nodes() - (graph.number_of_edges() / maximum_degree(graph)), decimals=5))

def Cvetkovic_bound(graph: nx.Graph) -> int:
    eigenvalues = np.real(nx.adjacency_spectrum(graph))
    eigenvalues = np.around(eigenvalues, decimals=5)
    negative_eigenvalues = np.count_nonzero(eigenvalues < 0.)
    positive_eigenvalues = np.count_nonzero(eigenvalues > 0.)
    zero_eigenvalues = np.count_nonzero(eigenvalues == 0.)
    return zero_eigenvalues + min(positive_eigenvalues, negative_eigenvalues)


def Konig_bound(graph: nx.Graph) -> int:
    maximal_matching_set = nx.maximal_matching(G)
    return graph.number_of_nodes() - len(maximal_matching_set)

def min_degree_bound(graph: nx.Graph) -> int:
    return graph.number_of_nodes() - minimum_degree(graph)

def Hensen_bound(graph: nx.Graph) -> int:
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    return int(1/2 + math.sqrt(1/4 + n**2 - n - 2*e))

def chromatic_number_bound(graph: nx.Graph) -> float:
    colors = set(nx.greedy_color(graph).values())
    return float(np.around(graph.number_of_nodes() / len(colors), decimals=5))

def lower_matchings_bound(graph: nx.Graph) -> int:
    maximal_matching_set = nx.maximal_matching(G)
    return graph.number_of_nodes() - 2 * len(maximal_matching_set)

def Turan_bound(graph: nx.Graph) -> float:
    sum_degrees = 2 * graph.number_of_edges()
    average_degree = sum_degrees / graph.number_of_nodes()
    return float(np.around(graph.number_of_nodes() / (1 + average_degree), decimals=5))

def Wei_bound(graph: nx.Graph) -> float:
    bound = 0
    for v in graph.nodes():
        bound += 1 / (1 + graph.degree[v])
    return float(np.around(bound, decimals=5))

def Wilf_bound(graph: nx.Graph) -> float:
    eigenvalues = np.real(nx.adjacency_spectrum(graph))
    eigenvalues = np.around(eigenvalues, decimals=5)
    return float(np.around(graph.number_of_nodes()/ (1 + np.max(eigenvalues)), decimals=5))

p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n = 10
independent_number_list = []
Kwok_bound_list = []
Cvetkovic_bound_list = []
Konig_bound_list = []
min_degree_bound_list = []
Hensen_bound_list = []
chromatic_number_bound_list = []
matching_bound_list = []
Turan_bound_list = []
Wei_bound_list = []
Wilf_bound_list = []
approx_independent_set_list = []
edges = []
time_list = []
alpha = 0
bound_Kwok = 0
bound_Cvetkovic = 0
bound_Konig = 0
bound_min_degree = 0
bound_Hensen = 0
bound_chromatic_number = 0
bound_matching_number = 0
bound_Turan = 0
bound_Wei = 0
bound_Wilf = 0
number_edges = 0
time_of_al = 0
approx_alpha = 0
iterations = 50
for p in p_list:
    for i in tqdm.tqdm(range(0, iterations)):
        G = nx.gnp_random_graph(n=n, p=p)
        if G.number_of_edges() == 0:
            G.add_edges_from([(0, 1)])
        start = default_timer()
        alpha += maximum_independent_set(G)
        finish = default_timer()
        time_of_al += finish - start
        approx_alpha += approx_independent_set(G)
        bound_Kwok += Kwok_bound(G)
        bound_Cvetkovic += Cvetkovic_bound(G)
        bound_Konig += Konig_bound(G)
        bound_min_degree += min_degree_bound(G)
        bound_Hensen += Hensen_bound(G)
        bound_chromatic_number += chromatic_number_bound(G)
        bound_matching_number += lower_matchings_bound(G)
        bound_Turan += Turan_bound(G)
        bound_Wei += Wei_bound(G)
        bound_Wilf += Wilf_bound(G)
        number_edges += G.number_of_edges()
    independent_number_list.append(alpha / iterations)
    Kwok_bound_list.append(bound_Kwok / iterations)
    Cvetkovic_bound_list.append(bound_Cvetkovic / iterations)
    Konig_bound_list.append(bound_Konig / iterations)
    min_degree_bound_list.append(bound_min_degree / iterations)
    Hensen_bound_list.append(bound_Hensen / iterations)
    chromatic_number_bound_list.append(bound_chromatic_number / iterations)
    matching_bound_list.append(bound_matching_number / iterations)
    Turan_bound_list.append(bound_Turan / iterations)
    Wei_bound_list.append(bound_Wei / iterations)
    Wilf_bound_list.append(bound_Wilf / iterations)
    edges.append(number_edges / iterations)
    time_list.append(time_of_al / iterations)
    approx_independent_set_list.append((approx_alpha / iterations))
    approx_alpha = 0
    alpha = 0
    bound_Kwok = 0
    bound_Cvetkovic = 0
    bound_Konig = 0
    bound_min_degree = 0
    bound_Hensen = 0
    bound_chromatic_number = 0
    bound_matching_number = 0
    bound_Turan = 0
    bound_Wei = 0
    bound_Wilf = 0
    number_edges = 0
    time_of_al = 0

print('Ok')



