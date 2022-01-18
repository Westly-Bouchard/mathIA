import random, math
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def __cmp__(self, other):
        return self.id == other.id

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def get_distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Graph:
    adjacencyMatrix = [[]]
    nodes = []

    def __init__(self, count):
        self.numNodes = count
        while len(self.nodes) != count:
            x = random.randint(0, 100)
            y = random.randint(0, 100)

            safe = True
            for node in self.nodes:
                if node.x == x or node.y == y:
                    safe = False
                    break

            if safe:
                self.nodes.append(Node(x, y, len(self.nodes)))

    def generate_adjacency_matrix(self):
        self.adjacencyMatrix = [[0 for i in range(self.numNodes)] for j in range(self.numNodes)]
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                if self.nodes[i] != self.nodes[j]:
                    self.adjacencyMatrix[i][j] = self.nodes[i].get_distance(self.nodes[j])


class Ant:
    def __init__(self, node):
        self.currentCount = 0
        self.tabu_list = [node]

    def __cmp__(self, other):
        return self.tabu_list[0] == other.tabu_list[0]

    def __repr__(self):
        return "Ant with starting node" + str(self.tabu_list[0])

    def reset(self):
        self.currentCount = 0
        self.tabu_list.clear()

    def update_count(self, count_to_add):
        self.currentCount += count_to_add

    def add_node(self, node):
        self.tabu_list.append(node)
        if len(self.tabu_list) > 1:
            delta = self.tabu_list[-2].get_distance(self.tabu_list[-1])
            self.update_count(delta)


class ACO:
    pheromone_matrix = [[]]
    ants = []

    def __init__(self, graph, alpha, beta, evap_const, starting_pheromone_const, num_iterations):
        self.pheromone_matrix = [[starting_pheromone_const for i in range(graph.numNodes)] for j in
                                 range(graph.numNodes)]
        self.alpha = alpha
        self.beta = beta
        self.evap_const = evap_const
        self.graph = graph
        self.num_iterations = num_iterations

    def solve(self):
        min_tour = 1e99
        evolution = []
        for i in range(self.num_iterations):
            self.place_ants()
            for i in range(self.graph.numNodes - 1):
                self.move_ants()
            for ant in self.ants:
                actual_count = ant.currentCount
                actual_count += ant.tabu_list[-1].get_distance(ant.tabu_list[0])
                if actual_count < min_tour:
                    min_tour = actual_count
            print(min_tour)
            evolution.append(min_tour)
            self.update_pheromone()
        return evolution

    def place_ants(self):
        self.ants.clear()
        while len(self.ants) != self.graph.numNodes:
            ant = Ant(self.graph.nodes[random.randint(0, self.graph.numNodes - 1)])
            safe = True
            for cmp_ant in self.ants:
                if ant.tabu_list[0].x == cmp_ant.tabu_list[0].x and ant.tabu_list[0].y == cmp_ant.tabu_list[0].y:
                    safe = False

            if safe:
                self.ants.append(ant)

    def move_ants(self):
        for ant in self.ants:
            # For all the nodes in the graph, if a node is not in the ant's tabu list, then the ant could possibly move
            # there
            possible_nodes = []
            desirability_sum = 0
            for node in self.graph.nodes:
                if node not in ant.tabu_list:
                    desirability = self.calculate_desirability(ant.tabu_list[-1], node)
                    possible_nodes.append([node, desirability])
                    desirability_sum += desirability

            for node in possible_nodes:
                node.append(node[1] / desirability_sum)

            # Now we actually need to choose a node and move the and there
            cumulative_sum = []
            for i in range(len(possible_nodes)):
                sum = 0
                for j in range(i, len(possible_nodes)):
                    sum += possible_nodes[j][-1]
                cumulative_sum.append(sum)

            choice = random.uniform(0, 1)
            node_choice = possible_nodes[-1][0]

            for k in range(len(cumulative_sum) - 1):
                if cumulative_sum[k] >= choice > cumulative_sum[k + 1]:
                    node_choice = possible_nodes[k][0]

            ant.add_node(node_choice)

    def calculate_desirability(self, current_node, possible_node):
        dist = current_node.get_distance(possible_node)
        # pheromone = 0
        pheromone = self.pheromone_matrix[self.graph.nodes.index(current_node) % 5][
            self.graph.nodes.index(possible_node) % 5]  # Need to figure out how to index into the pheromone matrix
        desirability = (pheromone ** self.alpha) * (1 / dist) ** self.beta
        return desirability

    def evaporate_pheromones(self):
        for i in range(len(self.pheromone_matrix)):
            for j in range(len(self.pheromone_matrix)):
                self.pheromone_matrix[i][j] = self.pheromone_matrix[i][j] * (1 - self.evap_const)

    def update_pheromone(self):
        self.evaporate_pheromones()

        for ant in self.ants:
            wrap_dist = ant.tabu_list[-1].get_distance(ant.tabu_list[0])
            total_dist = ant.currentCount + wrap_dist
            pheromone_amt = 1 / total_dist
            for i in range(len(ant.tabu_list) - 1):
                curr_node_idx = self.graph.nodes.index(ant.tabu_list[i])
                next_node_idx = self.graph.nodes.index(ant.tabu_list[i + 1])
                self.pheromone_matrix[curr_node_idx][next_node_idx] += pheromone_amt
                self.pheromone_matrix[next_node_idx][curr_node_idx] += pheromone_amt
            first_node_idx = self.graph.nodes.index(ant.tabu_list[0])
            last_node_idx = self.graph.nodes.index(ant.tabu_list[-1])
            self.pheromone_matrix[first_node_idx][last_node_idx] += pheromone_amt
            self.pheromone_matrix[last_node_idx][first_node_idx] += pheromone_amt


class Traditional:
    def __init__(self, graph):
        self.graph = deepcopy(graph)
        self.adjacency_matrix = self.graph.adjacencyMatrix

    def find_lower_bound(self):
        # This uses the deleted vertex algorithm
        # Delete a node
        deleted_node = self.graph.nodes.pop(random.randint(0, self.graph.numNodes - 1))
        # Generate MST
        mst_weight = 0
        mst = [self.graph.nodes[random.randint(0, len(self.graph.nodes) - 1)]]
        while len(mst) != len(self.graph.nodes):
            min_index = -1
            min_distance = 300
            for node in mst:
                for cmp_node in self.graph.nodes:
                    if cmp_node not in mst:
                        dist = node.get_distance(cmp_node)
                        if dist < min_distance:
                            min_distance = dist
                            min_index = self.graph.nodes.index(cmp_node)
            mst.append(self.graph.nodes[min_index])
            mst_weight += min_distance
        # Find the two edges of least weight that connect to the deleted vertex
        min = 300
        second_min = 300
        for node in self.graph.nodes:
            dist = deleted_node.get_distance(node)
            if dist < min:
                second_min = min
                min = dist
            elif dist < second_min:
                second_min = dist
        # Calculate final lower bound
        lower_bound = mst_weight + min + second_min
        return lower_bound

    def find_upper_bound(self):
        # Using the nearest neighbor algorithm
        path = [self.graph.nodes[random.randint(0, len(self.graph.nodes) - 1)]]
        while len(path) != self.graph.numNodes:
            possible_nodes = []
            # Find all the nodes that we could go to next
            for node in self.graph.nodes:
                if node not in path:
                    dist = path[-1].get_distance(node)
                    possible_nodes.append([node, dist])

            min = [Node(0, 0, 1e99), 500]
            # Find the node with the least distance
            for set in possible_nodes:
                if set[-1] < min[-1]:
                    min = set
            path.append(min[0])

        weight = 0
        for i in range(len(path) - 1):
            weight += path[i].get_distance(path[i + 1])
        weight += path[-1].get_distance(path[0])
        return weight


def main():
    num_nodes = 20
    num_iterations = 100
    graph = Graph(num_nodes)
    aco = ACO(graph, 0.2, 2, 0.65, 0.1, num_iterations)
    evolution = aco.solve()
    traditional_solver = Traditional(graph)
    l_bound = traditional_solver.find_lower_bound()
    u_bound = traditional_solver.find_upper_bound()
    print(l_bound)
    print(u_bound)
    lower_bound = [l_bound for i in range(num_iterations)]
    upper_bound = [u_bound for i in range(num_iterations)]
    x_axis = [i for i in range(num_iterations)]
    plt.plot(x_axis, evolution, label="ACO")
    plt.plot(x_axis, lower_bound, label="Lower Bound")
    plt.plot(x_axis, upper_bound, label="Upper Bound")
    plt.xlabel("# Of Iterations")
    plt.ylabel("Length of Solution")
    plt.legend()
    plt.title("Graph with " + str(num_nodes) + " Vertices")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
