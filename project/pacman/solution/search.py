from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # print("Start: %s" % (str(problem.startingState())))
    # print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    # print("Start's successors: %s" % (problem.successorStates(problem.startingState())))

    visited = []
    stack = Stack()
    neighbors = []

    stack.push(((problem.startingState()), []))

    while not stack.isEmpty():

        CurNode, path = stack.pop()

        if problem.isGoal(CurNode) is True:
            return path

        # print("Current Node: %s" % (str(CurNode)))
        # print("Is the start a goal?: %s" % (problem.isGoal(CurNode)))
        # print("Start's successors: %s" % (problem.successorStates(CurNode)))

        if CurNode in visited:
            continue

        visited.append(CurNode)
        neighbors = problem.successorStates(CurNode)

        for leaf in neighbors:
            if leaf[0] in visited:
                continue
            else:
                stack.push((leaf[0], path + [leaf[1]]))

    raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # print("Start: %s" % (str(problem.startingState())))
    # print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    # print("Start's successors: %s" % (problem.successorStates(problem.startingState())))

    visited = []
    queue = Queue()
    neighbors = []

    queue.push((problem.startingState(), []))
    # print(problem.startingState())
    while not queue.isEmpty():

        CurNode, path = queue.pop()

        # print(CurNode)
        # print(path)
        print(problem.isGoal(CurNode))
        if problem.isGoal(CurNode) is True:
            return path

        print("Current Node: %s" % (str(CurNode)))
        # print("Is the start a goal?: %s" % (problem.isGoal(CurNode)))
        # print("Start's successors: %s" % (problem.successorStates(CurNode)))

        if CurNode in visited:
            continue

        visited.append(CurNode)
        neighbors = problem.successorStates(CurNode)
        print(neighbors)

        for leaf in neighbors:
            if leaf[0] in visited:
                continue
            else:
                # print(leaf)
                queue.push((leaf[0], path + [leaf[1]]))

    raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    visited = []
    queue = PriorityQueue()
    neighbors = []

    queue.push(((problem.startingState()), []), 0)

    while not queue.isEmpty():

        CurNode, path = queue.pop()

        if problem.isGoal(CurNode) is True:
            return path

        # print("Current Node: %s" % (str(CurNode)))
        # print("Is the start a goal?: %s" % (problem.isGoal(CurNode)))
        # print("Start's successors: %s" % (problem.successorStates(CurNode)))

        if CurNode in visited:
            continue

        visited.append(CurNode)
        neighbors = problem.successorStates(CurNode)

        for leaf in neighbors:
            if leaf[0] in visited:
                continue
            else:
                total_cost = problem.actionsCost(path + [leaf[1]]) + leaf[2]
                queue.push((leaf[0], path + [leaf[1]]), total_cost)

    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    visited = []
    queue = PriorityQueue()
    neighbors = []

    queue.push(((problem.startingState()), []), 0)

    while not queue.isEmpty():

        CurNode, path = queue.pop()

        if problem.isGoal(CurNode) is True:
            return path

        # print("Current Node: %s" % (str(CurNode)))
        # print("Is the start a goal?: %s" % (problem.isGoal(CurNode)))
        # print("Start's successors: %s" % (problem.successorStates(CurNode)))

        if CurNode in visited:
            continue

        visited.append(CurNode)
        neighbors = problem.successorStates(CurNode)

        for leaf in neighbors:
            if leaf[0] in visited:
                continue
            else:
                total_cost = problem.actionsCost(path + [leaf[1]]) + heuristic(leaf[0], problem)
                queue.push((leaf[0], path + [leaf[1]]), total_cost)

    raise NotImplementedError()
