from __future__ import print_function
from ortools.linear_solver import pywraplp
import json
import argparse

MAX_FUSABLE_BONUS = 0.9


def get_args():
    parser = argparse.ArgumentParser(description='Construct and solve ILP')
    parser.add_argument('--time_lim_sec', type=int, default=10, metavar='N',
        help='Time limit in seconds (default: 10)')
    parser.add_argument('--order_var_int', action='store_true', default=False,
        help='Use integer variable for t (variable for topological order)')
    parser.add_argument('--eclass_constraint', action='store_true', default=False,
        help='Add constraint that each eclass sum to at most 1')
    parser.add_argument('--no_order', action='store_true', default=False,
        help='No ordering constraints')
    parser.add_argument('--num_thread', type=int, default=1, metavar='N',
        help='Number of thread for the solver (default: 1)')
    parser.add_argument('--print_solution', action='store_true', default=False,
        help='To print out solution')
    parser.add_argument('--initialize', action='store_true', default=False,
        help='initialize with greedy solution')
    parser.add_argument('--fusion-costs', action='store_true', default=False,
        help='model costs of kernel fusion')

    return parser.parse_args()


def main():
    # Parse arguments
    args = get_args()

    # Load ILP data
    # - costs: cost for each node
    # - e: e[m] is the set of nodes within eclass m
    # - h: h[i] is the set of children eclasses for node i
    # - g: g(i) gives the eclass of node i
    with open('./tmp/ilp_data.json') as f:
        data = json.load(f)

    costs = data['cost_i']
    fus = data['fus_i']
    e = data['e_m']
    h = data['h_i']
    g = data['g_i']
    root_m = data['root_m']
    blacklist_i = data['blacklist_i']
    num_nodes = len(costs)
    num_classes = len(e)

    epsilon = 1 / (10 * num_classes)
    if args.order_var_int:
        A = num_classes
    else:
        A = 2

    # Create solver
    solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    if args.num_thread != 1:
        print("Set number of threads to {}".format(args.num_thread))
        solver.SetNumThreads(args.num_thread)
        
    if args.time_lim_sec > 0:
        print("Set time limit to {} seconds".format(args.time_lim_sec))
        solver.SetTimeLimit(args.time_lim_sec * 1000)

    # Define variables
    # - x: an integer variable for each node. x[i] = 1 means node i is picked
    # - t: a variable for each eclass reflecting topological ordering. This is to ensure the 
    #      extracted graph has a valid topological order, thus does not contain cycles
    x = {}
    for j in range(num_nodes):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)

    t = {}
    if args.order_var_int:
        print("Use int var for order")
        for j in range(num_classes):
            t[j] = solver.IntVar(0, num_classes-1, 't[%i]' % j)
    else:
        for j in range(num_classes):
            t[j] = solver.NumVar(0.0, 1.0, 't[%i]' % j)

    print('Number of variables =', solver.NumVariables())

    # Define constraints
    # Root
    solver.Add(sum([x[j] for j in e[root_m]]) == 1)

    if args.eclass_constraint:
        # eclass_constraints are optional because in most cases, the solution that minimizes
        # the total cost will only contain 1 picked node for each picked eclass, so we don't 
        # have to explicity include this.
        print("Add eclass constraints")
        for m in range(num_classes):
            solver.Add(sum([x[j] for j in e[m]]) <= 1)
    
    for i in range(num_nodes):
        for m in h[i]:
            # Children
            solver.Add(sum([x[j] for j in e[m]]) - x[i] >= 0)
            # Order
            # We only need to add ordering costraints when there are potentially cycles in the
            # extracted graph. If the EGraph itself does not contain cycles, then we don't need
            # these constraints
            if not args.no_order:
                if args.order_var_int:
                    solver.Add(t[g[i]] - t[m] + A * (1 - x[i]) >= 1)
                else:
                    solver.Add(t[g[i]] - t[m] - epsilon + A * (1 - x[i]) >= 0)

    # Blacklist constraints
    for j in blacklist_i:
        solver.Add(x[j] == 0)

    # fusion shenanigans
    class_has_fusable_node = {}
    for j in range(num_classes):
        class_has_fusable_node[j] = solver.IntVar(0, 1, 'class_has_fusable_node[%i]' % j)

    for i in range(num_nodes):
        if not fus[i]:
            # a class is fusable only if no un-fusable nodes are chosen
            solver.Add(class_has_fusable_node[g[i]] <= 1 - x[i])
    
    # note that both [node_count_fusable_children] and [/_if_picked] 
    #  are actually upper bounds for the relevant counts, 
    #  but they end up maximised through the [fus_expr] objective
    node_count_fusable_children = {}
    for i in range(num_nodes):
        node_count_fusable_children[i] = solver.IntVar(0, len(h[i]), 'node_count_fusable_children[%i]' % j)
        if not fus[i]:
            # if the node itself cannot fuse, it can never fuse
            solver.Add(node_count_fusable_children[i] <= 0)
        # count children that can fuse
        solver.Add(node_count_fusable_children[i] <= sum(class_has_fusable_node[m] for m in h[i]))

    # [node_count_fusable_children] if [x = 1], else 0
    node_count_fusable_children_if_picked = {}
    for i in range(num_nodes):
        node_count_fusable_children_if_picked[i] = solver.IntVar(0, len(h[i]), 'node_count_fusable_children_if_picked[%i]' % j)
        solver.Add(node_count_fusable_children_if_picked[i] <= x[i] * len(h[i]))
        solver.Add(node_count_fusable_children_if_picked[i] <= node_count_fusable_children[i])

    # Define objective
    obj_expr = [costs[j] * x[j] for j in range(num_nodes)]
    fus_expr = [
        # subtract fraction of consumer runtime
        # equal to the ratio of fusable producers for that consumer
        # scaling from 0 to MAX_FUSABLE_BONUS 
        # (MAX_FUSABLE_BONUS = 1 means: 
        #   if all producers are fusable contribute 0 in total to objective)
        -1 * MAX_FUSABLE_BONUS * (1 / max(1, len(h[i]))) 
           * costs[j] * node_count_fusable_children_if_picked[j] 
        for j in range(num_nodes)
    ]

    if args.fusion_costs:
        objective = sum(obj_expr) + sum(fus_expr)        
    else:
        objective = sum(obj_expr)

    solver.Minimize(objective)

    # Set initial solutions
    if args.initialize:
        print("Initialize with greedy")
        with open('./tmp/init_sol.json') as f:
            sol_data = json.load(f)

        i_list = sol_data['i_list']
        m_list = sol_data['m_list']

        i_var_list = [x[i] for i in range(num_nodes)]
        i_init_val_list = [0 for i in range(num_nodes)]
        for i in i_list:
            i_init_val_list[i] = 1

        t_var_list = [t[m] for m in range(num_classes)]
        t_init_val_list = [0 for m in range(num_classes)]
        num_picked = len(m_list)
        gap = 1 / num_picked
        count = 0
        for m in m_list:
            if args.order_var_int:
                t_init_val_list[m] = count
            else:
                t_init_val_list[m] = count * gap
            count += 1

        solver.SetHint(i_var_list + t_var_list, i_init_val_list + t_init_val_list)

    # Solve
    status = solver.Solve()
    solve_time = solver.wall_time()
    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        print('Problem solved in %f milliseconds' % solve_time)
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        if args.print_solution:
            for j in range(num_nodes):
                print(x[j].name(), ' = ', x[j].solution_value())
            for j in range(num_classes):
                print(t[j].name(), ' = ', t[j].solution_value())
                
    else:
        print('The problem does not have an optimal solution.')
        print(status)
        print('Objective value =', solver.Objective().Value())
        print('Problem solved in %f milliseconds' % solve_time)
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

    # Store results
    solved_x = [int(x[j].solution_value()) for j in range(num_nodes)]
    solved_x_f = [int(node_count_fusable_children_if_picked[j].solution_value()) for j in range(num_nodes)]
    result_dict = {}
    result_dict["solved_x"] = solved_x
    result_dict["solved_x_f"] = solved_x_f
    result_dict["cost"] = solver.Objective().Value()
    result_dict["time"] = solve_time / 1000
    with open('./tmp/solved.json', 'w') as f:
        json.dump(result_dict, f)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
