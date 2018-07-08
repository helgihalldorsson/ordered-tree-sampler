import random
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout


def _number_of_children(xi, norm_const=1):
    """
    Randomly selects the number of children using the discrete density function xi.
    :param xi: List or lambda function such that xi[i] is the probability of i children, i>=0.
    :param norm_const: Normalizing constant, sum of all positive values in xi.
    :return: Randomly selected integer following the discrete density function xi.
    """
    # Instead of dividing each probability by the normalizing constant, we 'cheat' by scaling the interval.
    r = random.uniform(0, norm_const)
    i = 0
    is_callable = callable(xi)
    while r > 0:
        if is_callable:
            p = xi(i)
        else:
            if i >= len(xi):
                return 0
            p = xi[i]
        if r <= p:
            return i
        r -= p
        i += 1
    # This point should never be reached, as sum(xi) <= norm_const should be true.
    return 0


def _get_normalizing_constant(xi, max_children=10000):
    """
    Calculates the sum of xi[0,1,...,max_children]. This can be used to convert weights into probability.
    :param xi: List or lambda function such that xi[i] / sum(xi[0,...,max_children]) is the probability of a node
    getting i children.
    :param max_children: Maximum number of children a node can have. This parameter is ignored if xi is not callable.
    :return: Sum of xi[0,1,...,len(xi)-1] or sum of xi(0,1,...,max_children) if xi is callable.
    """
    if callable(xi):
        if xi(0) <= 0:
            raise ValueError('The probability of getting 0 children must be positive.')
        return sum([xi(i) for i in range(max_children+1)])

    if xi[0] <= 0:
        raise ValueError('The probability of getting 0 children must be positive.')
    return sum(xi)


def _generate_random_walk(steps, xi, max_iter=1000, max_children=10000, adjustable_size=False, max_adjustment=0.05):
    """
    Generates a list of each step of a random walk, starting at 1 and taking n random steps following the
    probability density xi, ending the walk at 0. It will retry until it ends at 0 or reaches max_iter attemts.
    :param steps: Number of steps to take.
    :param xi: Discrete density function such that x[i] is the probability of taking i-1 steps up.
    :param max_iter: Maximum number of attempts before stopping.
    :param adjustable_size: Generates a tree with approximately tree_size nodes instead of exactly, to guarantee
    a success.
    :param max_adjustment: Maximum percentage that the length of the walk can vary in case it is unsuccessful for the
    given number of steps. Only used if adjustable_size = True.
    :return: List with n+1 elements, containing a random walk with n steps, starting at 1 and ending at 0.
    """
    attempts = 0
    norm_const = _get_normalizing_constant(xi, max_children)
    min_walk = None
    while attempts < max_iter:
        walk = [1]
        for i in range(0, steps):
            walk.append(walk[i] - 1 + _number_of_children(xi, norm_const))

        if walk[steps] == 0:
            return walk

        # Keeps track of the walk that ends closest to 0.
        if min_walk is None:
            min_walk = walk
        else:
            if abs(walk[steps]) < abs(min_walk[steps]):
                min_walk = walk

        attempts += 1

    if adjustable_size:
        delta = int(steps * max_adjustment)

        min_alt_end = None
        index = 1
        for k in range(int(max_iter/25)):
            for j in range(delta):
                alt_end = [min_walk[steps - delta]]
                if min_alt_end is None:
                    min_alt_end = [min_walk[steps - delta]]
                for j in range(2*index):
                    alt_end.append(alt_end[j] - 1 + _number_of_children(xi, norm_const))
                    if alt_end[len(alt_end)-1] == 0:
                        break
                if alt_end[len(alt_end)-1] == 0 and abs(len(alt_end) - delta) < abs(len(min_alt_end) - delta):
                    min_alt_end = alt_end
                    index = j

                if min_alt_end[len(min_alt_end)-1] == 0:
                    break

        return min_walk[0:(steps - index + 1)] + min_alt_end[1:]

    raise ValueError('Reached max_iter=' + str(max_iter) + ' attempts at generating a random walk without success.')


def _get_queue_size_walk(steps, xi, max_iter=1000, max_children=10000, adjustable_size=False):
    """
    This function generates a walk describing the number of nodes left in a queue of nodes at each step.
    The walk starts at 1 (queue contains the root). Each node is removed and replaced with i child nodes,
    following the density function xi. The walk is strictly positive until it ends with 0.
    :param steps: Number of steps.
    :param xi: Discrete density function.
    :param max_iter: Maximum number of attempts before stopping.
    :param max_children: Maximum number of children a node can have.
    :param adjustable_size: Generates a tree with approximately tree_size nodes instead of exactly, to guarantee
    a success.
    :return: A walk starting at 1 and ending at 0, strictly positive between.
    """
    walk = _generate_random_walk(steps, xi, max_iter=max_iter, max_children=max_children,
                                 adjustable_size=adjustable_size)
    min_value = min(walk[0:(steps - 1)])

    # The walk describes the number of nodes left in a queue where each node is removed and given i children, which
    # are all added to the queue. Starts at 1 (the root) and must remain strictly positive until it reaches 0 in the
    # final step. Because of this, we must check if goes below 1 at any point.

    # If the path goes below 1 before the end, the walk can be 'cut' at the minimum value, swapping the parts before
    # and after and connecting the ends. This results in a walk that meets the conditions without affecting the
    # probability.

    if min_value < 1:
        min_index = walk.index(min_value)
        walk_start = [step - min_value for step in walk[1:min_index + 1]]
        walk_end = [step + 1 - min_value for step in walk[min_index:]]
        walk = walk_end + walk_start

    return walk


def _get_node_degrees(walk):
    """
    Converts a random walk to the number of children each node has, following the pre-order of the tree.
    :param walk: A list of integers, starting at 1 and ending at 0, remaining strictly positive between.
    :return: The number of children each node has, ordered by the pre-order of the tree.
    """
    # The walk represents a queue of nodes that have not been given children. Thus, walk[i] - walk[i-1] +1
    # is the number of children that node i receives.
    seq = [walk[1]]
    for i in range(2, len(walk)):
        seq.append(walk[i] - walk[i - 1] + 1)
    return seq


def _get_expected_value(xi, max_children=10000, nc=1):
    if callable(xi):
        return sum([i*xi(i) for i in range(1, max_children+1)])/nc
    else:
        return sum([i*xi[i] for i in range(len(xi))])/nc


def adjust_density_distribution(xi, max_children=10000):
    """
    The expected value of xi must be approximately 1 to make sure the walk can end at 0.
    :param xi: Discrete density distribution.
    :param max_children: Maximum number of children.
    :return: Adjusted xi distribution by normalizing and adjusting the probability of 0 to shift the expected value
    to 1.
    """
    norm_const = _get_normalizing_constant(xi, max_children)
    expected_value = _get_expected_value(xi, max_children, norm_const)
    if abs(expected_value - 1) < 0.00000000001:
        return xi
    if callable(xi):
        return lambda x: xi(x) / (expected_value * norm_const) if x > 0 \
            else (xi[0]/norm_const + expected_value - 1)/expected_value
    else:
        xi_adjusted = [p/(norm_const * expected_value) for p in xi]
        xi_adjusted[0] += + (expected_value - 1)/expected_value
        return xi_adjusted


def analyse_density_function(steps, xi, iter=25000, max_children=10000):
    """
    :param steps: Number of steps to take.
    :param xi: Discrete density function such that x[i] is the probability of taking i-1 steps up.
    :param iter: Number of trials.
    :param max_children: Maximum number of children a node can have.
    :return: Expected and variance of the end of iter walks.
    """
    counter = {}
    xi = adjust_density_distribution(xi, max_children)
    for i in range(iter):
        walk = [1]
        for j in range(0, steps):
            walk.append(walk[j] - 1 + _number_of_children(xi, 1))

        if not walk[steps] in counter:
            counter[walk[steps]] = 0
        counter[walk[steps]] += 1

    occurances = 0
    if 0 in counter:
        occurances = counter[0]
    print("Occurances of 0: " + str(occurances) + " / " + str(iter))
    plt.bar(list(counter.keys()), list(counter.values()))
    plt.show()

    expected_value = sum([k * counter[k] / iter for k in counter.keys()])
    variance = sum([k * k * counter[k] / iter for k in counter.keys()])
    return expected_value, variance


def plot_tree(tree):
    pos = graphviz_layout(tree)
    nx.draw_networkx_nodes(tree, pos, node_size=1)
    nx.draw_networkx_edges(tree, pos, alpha=0.1)
    plt.show()


def sample_tree(tree_size, xi=lambda x: pow(2, - x - 1), adjustable_size=False, max_iter=1000, max_children=10000):
    """
    Generates a random tree using the Graph object in the NetworkX package.
    The default probability distribution (xi) is the uniform distribution of randomly selecting an ordered
    tree from the set of all trees of size tree_size.
    :param tree_size: Number of nodes in the tree.
    :param xi: List or lambda function. xi[i]/sum(xi[0,1,...,max_children]) or xi[i]/sum(xi[0,1,...,len(xi)])
    is the probability of a node getting i children.
    :param adjustable_size: Generates a tree with approximately tree_size nodes instead of exactly, to guarantee
    a success.
    :param max_iter: Maximum number of attempts before stopping.
    :param max_children: Maximum number of children a node can have. This is ignored if xi is not callable.
    :return: A randomly selected ordered tree and the number of nodes.
    """
    xi = adjust_density_distribution(xi, max_children)
    walk = _get_queue_size_walk(tree_size, xi, max_iter, max_children, adjustable_size)
    node_degrees = _get_node_degrees(walk)

    tree = nx.Graph()
    for i in range(0, len(node_degrees)):
        tree.add_node(i)

    n = 0
    stack = [0]

    for i in range(0, len(node_degrees)):
        current_node = stack[len(stack) - 1]
        stack = stack[:len(stack) - 1]

        if node_degrees[i] > 0:
            for j in range(0, node_degrees[i]):
                n += 1
                stack += [n]
                tree.add_edge(current_node, n)

    if adjustable_size:
        return tree, len(node_degrees)
    return tree
