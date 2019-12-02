"""
===================
create_expressions.py
===================
Create a dataset of logic expression trees
"""



import itertools
import pickle as pkl
import copy
import numpy as np


class LogicNode(object):

    def __init__(self, parent):
        self.parent = parent

    def set_parent(self, parent):
        self.parent = parent


class TrueNode(LogicNode):

    def __init__(self, parent):
        LogicNode.__init__(self, parent)
        self.token = "T"

    def copy(self):
        return TrueNode(self.parent)

    def new_var_p(self):
        ornode = OrNode(self.parent)
        pnode = PNode(ornode)
        ornode.set_lr(self, pnode)
        return ornode

    def new_var_q(self):
        ornode = OrNode(self.parent)
        pnode = QNode(ornode)
        ornode.set_lr(self, pnode)
        return ornode

    def new_var_r(self):
        ornode = OrNode(self.parent)
        pnode = RNode(ornode)
        ornode.set_lr(self, pnode)
        return ornode

    def new_var_notp(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        pnode = PNode(notnode)
        notnode.set_arg(pnode)
        ornode.set_lr(self, notnode)
        return ornode

    def new_var_notq(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        qnode = QNode(notnode)
        notnode.set_arg(qnode)
        ornode.set_lr(self, notnode)
        return ornode

    def new_var_notr(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        rnode = RNode(notnode)
        notnode.set_arg(rnode)
        ornode.set_lr(self, notnode)
        return ornode

    def new_tautology_p(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        pnode1 = PNode(notnode)
        notnode.set_arg(pnode1)
        pnode2 = PNode(ornode)
        ornode.set_lr(notnode, pnode2)
        return ornode

    def new_tautology_q(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        qnode1 = QNode(notnode)
        notnode.set_arg(qnode1)
        qnode2 = QNode(ornode)
        ornode.set_lr(notnode, qnode2)
        return ornode

    def new_tautology_r(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        rnode1 = RNode(notnode)
        notnode.set_arg(rnode1)
        rnode2 = RNode(ornode)
        ornode.set_lr(notnode, rnode2)
        return ornode


    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.new_var_p(), 1), \
                    (self.new_var_q(), 2), \
                    (self.new_var_r(), 3), \
                    (self.new_var_notp(), 4), \
                    (self.new_var_notq(), 5), \
                    (self.new_var_notr(), 6), \
                    (self.new_tautology_p(), 7), \
                    (self.new_tautology_q(), 8), \
                    (self.new_tautology_r(), 9), \
                    (self.identity(), 10) \
                    ]


        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes




class FalseNode(LogicNode):

    def __init__(self, parent):
        LogicNode.__init__(self, parent)
        self.token = "F"

    def copy(self):
        return FalseNode(self.parent)

    def neg_f(self):
        notnode = NotNode(self.parent)
        truenode = TrueNode(notnode)
        notnode.set_arg(truenode)
        return notnode

    def new_var_p_f(self):
        andnode = AndNode(self.parent)
        pnode = PNode(andnode)
        andnode.set_lr(self, pnode)
        return andnode

    def new_var_q_f(self):
        andnode = AndNode(self.parent)
        pnode = QNode(andnode)
        andnode.set_lr(self, pnode)
        return andnode

    def new_var_r_f(self):
        andnode = AndNode(self.parent)
        pnode = RNode(andnode)
        andnode.set_lr(self, pnode)
        return andnode

    def new_var_notp_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        pnode = PNode(notnode)
        notnode.set_arg(pnode)
        andnode.set_lr(self, notnode)
        return andnode

    def new_var_notq_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        qnode = QNode(notnode)
        notnode.set_arg(qnode)
        andnode.set_lr(self, notnode)
        return andnode

    def new_var_notr_f(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        rnode = RNode(notnode)
        notnode.set_arg(rnode)
        andnode.set_lr(self, notnode)
        return andnode

    def new_fallacy_p(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        pnode1 = PNode(notnode)
        notnode.set_arg(pnode1)
        pnode2 = PNode(andnode)
        andnode.set_lr(notnode, pnode2)
        return andnode

    def new_fallacy_q(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        qnode1 = QNode(notnode)
        notnode.set_arg(qnode1)
        qnode2 = QNode(andnode)
        andnode.set_lr(notnode, qnode2)
        return andnode

    def new_fallacy_r(self):
        andnode = AndNode(self.parent)
        notnode = NotNode(andnode)
        rnode1 = RNode(notnode)
        notnode.set_arg(rnode1)
        rnode2 = RNode(andnode)
        andnode.set_lr(notnode, rnode2)
        return andnode

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.new_var_p_f(), 31), \
                    (self.new_var_q_f(), 32), \
                    (self.new_var_r_f(), 33), \
                    (self.new_var_notp_f(), 34), \
                    (self.new_var_notq_f(), 35), \
                    (self.new_var_notr_f(), 36), \
                    (self.new_fallacy_p(), 37), \
                    (self.new_fallacy_q(), 38), \
                    (self.new_fallacy_r(), 39), \
                    (self.neg_f(), 40), \
                    (self.identity(), 11) \
                    ]


        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class PNode(LogicNode):

    def __init__(self, parent):
        LogicNode.__init__(self, parent)
        self.token = "p"

    def copy(self):
        return PNode(self.parent)

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.identity(), 12)]
        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes



class QNode(LogicNode):

    def __init__(self, parent):
        LogicNode.__init__(self, parent)
        self.token = "q"

    def copy(self):
        return QNode(self.parent)

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.identity(), 13)]
        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class RNode(LogicNode):

    def __init__(self, parent):
        LogicNode.__init__(self, parent)
        self.token = "r"

    def copy(self):
        return RNode(self.parent)

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.identity(), 14)]
        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes


class AndNode(LogicNode):

    def __init__(self, parent, left=None, right=None):
        LogicNode.__init__(self, parent)
        self.token = "∧"
        self.left = left
        self.right = right

    def set_lr(self, l, r):
        l.parent = self
        r.parent = self
        self.left = l
        self.right = r

    def copy(self):
        andnode = AndNode(self.parent)
        andnode.set_lr(self.left.copy(), self.right.copy())
        return andnode

    def commutative1(self):
        andnode = AndNode(self.parent)
        andnode.set_lr(self.right.copy(), self.left.copy())
        return andnode

    def associative3(self):
        if isinstance(self.right, AndNode):
            topandnode = AndNode(self.parent)
            bottomandnode = AndNode(topandnode)
            bottomandnode.set_lr(self.left.copy(), self.right.left.copy())
            topandnode.set_lr(bottomandnode, self.right.right.copy())
            return topandnode

        else:
            return False

    def associative4(self):
        if isinstance(self.left, AndNode):
            topandnode = AndNode(self.parent)
            bottomandnode = AndNode(topandnode)
            bottomandnode.set_lr(self.left.right.copy(), self.right.copy())
            topandnode.set_lr(self.left.left.copy(), bottomandnode)
            return topandnode
        else:
            return False

    def demorgan4(self):
        if isinstance(self.left, NotNode) and isinstance(self.right, NotNode):
            l = self.left.arg.copy()
            r = self.right.arg.copy()
            notnode = NotNode(self.parent)
            ornode = OrNode(notnode)
            ornode.set_lr(l, r)
            notnode.set_arg(ornode)
            return notnode
        else:
            return False

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.commutative1(), 15), \
                    (self.associative3(), 16), \
                    (self.associative4(), 17), \
                    (self.demorgan4(), 18), \
                    (self.identity(), 19) \
                    ]

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes




class OrNode(LogicNode):

    def __init__(self, parent, left=None, right=None):
        LogicNode.__init__(self, parent)
        self.token = "∨"
        self.left = left
        self.right = right

    def set_lr(self, l, r):
        l.parent = self
        r.parent = self
        self.left = l
        self.right = r


    def copy(self):
        ornode = OrNode(self.parent)
        ornode.set_lr(self.left.copy(), self.right.copy())
        return ornode


    def commutative2(self):
        ornode = OrNode(self.parent)
        ornode.set_lr(self.right.copy(), self.left.copy())
        return ornode


    def associative1(self):
        if isinstance(self.right, OrNode):
            topornode = OrNode(self.parent)
            bottomornode = OrNode(topornode)
            bottomornode.set_lr(self.left.copy(), self.right.left.copy())
            topornode.set_lr(bottomornode, self.right.right.copy())
            return topornode
        else:
            return False

    def associative2(self):
        if isinstance(self.left, OrNode):
            topornode = OrNode(self.parent)
            bottomornode = OrNode(topornode)
            bottomornode.set_lr(self.left.right.copy(), self.right.copy())
            topornode.set_lr(self.left.left.copy(), bottomornode)
            return topornode
        else:
            return False

    def logic_equiv(self):
        if (isinstance(self.left, NotNode) and not isinstance(self.right, NotNode)):
            implies = ImplicationNode(self.parent)
            condition = self.left.arg.copy()
            implication = self.right.copy()
            implies.set_lr(condition, implication)
            return implies
        elif (isinstance(self.right, NotNode) and not isinstance(self.left, NotNode)):
            implies = ImplicationNode(self.parent)
            condition = self.right.arg.copy()
            implication = self.left.copy()
            implies.set_lr(condition, implication)
            return implies
        else:
            return False

    def demorgan3(self):
        if isinstance(self.left, NotNode) and isinstance(self.right, NotNode):
            l = self.left.arg.copy()
            r = self.right.arg.copy()
            notnode = NotNode(self.parent)
            andnode = AndNode(notnode)
            andnode.set_lr(l, r)
            notnode.set_arg(andnode)
            return notnode
        else:
            return False

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.commutative2(), 20), \
                    (self.associative1(), 21), \
                    (self.associative2(), 22), \
                    (self.logic_equiv(), 23), \
                    (self.demorgan3(), 24), \
                    (self.identity(), 25) \
                    ]

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes



class ImplicationNode(LogicNode):

    def __init__(self, parent, left=None, right=None):
        LogicNode.__init__(self, parent)
        self.token = "→"
        self.left = left
        self.right = right

    def set_lr(self, l, r):
        l.parent = self
        r.parent = self
        self.left = l
        self.right = r

    def copy(self):
        return ImplicationNode(self.parent, self.left.copy(), self.right.copy())

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def to_or(self):
        ornode = OrNode(self.parent)
        notnode = NotNode(ornode)
        notnode.set_arg(self.left)
        ornode.set_lr(notnode, self.right)
        return ornode


    def do_ops(self):

        new_nodes = [(self.to_or(), 26), \
                    (self.identity(), 27) \
                    ]

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes



class NotNode(LogicNode):

    def __init__(self, parent, arg=None):
        LogicNode.__init__(self, parent)
        self.token = "~"
        self.arg = arg

    def set_arg(self, ar):
        ar.parent = self
        self.arg = ar


    def copy(self):
        return NotNode(self.parent, self.arg.copy())

    def demorgan1(self):
        if isinstance(self.arg, AndNode):
            andnode = self.arg
            ornode = OrNode(self.parent)
            ornode.set_lr(NotNode(ornode, andnode.left.copy()), NotNode(ornode, andnode.right.copy()))
            return ornode
        else:
            return False

    def demorgan2(self):
        if isinstance(self.arg, OrNode):
            ornode = self.arg
            andnode = AndNode(self.parent)
            andnode.set_lr(NotNode(andnode, ornode.left.copy()), NotNode(andnode, ornode.right.copy()))
            return andnode
        else:
            return False

    def identity(self):
        andnode = AndNode(self.parent)
        truenode = TrueNode(andnode)
        andnode.set_lr(self, truenode)
        return andnode

    def do_ops(self):

        new_nodes = [(self.demorgan1(), 28), \
                    (self.demorgan2(), 29), \
                    (self.identity(), 30) \
                    ]

        new_nodes = [node for node in new_nodes if node[0] is not False]
        return new_nodes




class LogicTree():


    def __init__(self, postfix_tree=None):
        if postfix_tree != None:
            self.construct_tree(postfix_tree)
            self.computed_ops = 0
        else:
            self.root = None
            self.computed_ops = 0

        self.most_recent_op = 0

    def set_root(self, node):
        node.parent = self
        self.root = node

    def construct_tree(self, str):
        stack = []

        for char in str:

            if char == "p":
                stack.append(PNode(None))
            elif char == "q":
                stack.append(QNode(None))
            elif char == "r":
                stack.append(RNode(None))
            elif char == "T":
                stack.append(TrueNode(None))
            elif char == "F":
                stack.append(FalseNode(None))
            elif char == "~":
                arg = stack.pop()
                notnode = NotNode(None)
                notnode.set_arg(arg)
                stack.append(notnode)
            elif char == "∧" or char == "^":
                andnode = AndNode(None)
                right = stack.pop()
                left = stack.pop()
                andnode.set_lr(left, right)
                stack.append(andnode)
            elif char == "∨" or char == "v":
                ornode = OrNode(None)
                right = stack.pop()
                left = stack.pop()
                ornode.set_lr(left, right)
                stack.append(ornode)
            elif char == "→":
                imp_node = ImplicationNode(None)
                right = stack.pop()
                left = stack.pop()
                imp_node.set_lr(left, right)
                stack.append(imp_node)

        self.set_root(stack.pop())


    def set_computed_ops(self, ops):
        self.computed_ops = ops

    def copy(self):
        new_tree = LogicTree()
        new_tree.set_root(self.root.copy())
        new_tree.set_computed_ops(self.computed_ops)
        return new_tree

    def deep_ops(self):

        original_tree = self.copy()
        self.computed_ops += 1

        new_trees = []
        prohibited_ops = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 19, 25, 27, 30]

        # fully_prohibited_ops = [15,20]

        def deep_ops_helper(node, full_tree):
            if isinstance(node, PNode) or \
                isinstance(node, QNode) or \
                isinstance(node, RNode):
                pass
                # MAY NEED TO HAVE A BASE CASE HERE...

            # IF THE NODE IS THE LOGICTREE
            elif isinstance(node, LogicTree):
                original_root = node.root.copy()
                new_nodes = node.root.do_ops()
                # new_nodes = [node for node in new_nodes if node[1] not in fully_prohibited_ops]
                if self.computed_ops >=3:
                    new_nodes = [node for node in new_nodes if node[1] not in prohibited_ops]
                for new_node in new_nodes:
                    node.set_root(new_node[0])
                    new_trees.append((full_tree.copy(), new_node[1]))
                node.set_root(original_root)
                deep_ops_helper(node.root, full_tree)



            elif isinstance(node, NotNode):
                original_arg = node.arg.copy()
                new_nodes = node.arg.do_ops()
                if self.computed_ops >=3:
                    new_nodes = [node for node in new_nodes if node[1] not in prohibited_ops]
                for new_node in new_nodes:
                    node.set_arg(new_node[0])
                    new_trees.append((full_tree.copy(), new_node[1]))
                node.set_arg(original_arg)
                deep_ops_helper(node.arg, full_tree)

            elif isinstance(node, AndNode) or isinstance(node, OrNode) or isinstance(node, ImplicationNode):
                original_left = node.left.copy()
                new_left_nodes = node.left.do_ops()
                if self.computed_ops >=3:
                    new_left_nodes = [node for node in new_left_nodes if node[1] not in prohibited_ops]
                for new_node in new_left_nodes:
                    node.set_lr(new_node[0], node.right)
                    new_trees.append((full_tree.copy(), new_node[1]))
                node.set_lr(original_left, node.right)

                original_right = node.right.copy()
                new_right_nodes = node.right.do_ops()
                if self.computed_ops >=3:
                    new_right_nodes = [node for node in new_right_nodes if node[1] not in prohibited_ops]
                for new_node in new_right_nodes:
                    node.set_lr(node.left, new_node[0])
                    new_trees.append((full_tree.copy(), new_node[1]))
                node.set_lr(node.left, original_right)

                deep_ops_helper(node.left, full_tree)
                deep_ops_helper(node.right, full_tree)



        deep_ops_helper(self, self)
        return new_trees


    def parse_tree(self):

        def parse_helper(node):
            if isinstance(node, PNode) or \
                isinstance(node, QNode) or \
                isinstance(node, RNode) or \
                isinstance(node, TrueNode) or \
                isinstance(node, FalseNode):
                return node.token
            elif isinstance(node, NotNode):
                if isinstance(node.arg, PNode) or \
                    isinstance(node.arg, QNode) or \
                    isinstance(node.arg, RNode) or \
                    isinstance(node.arg, TrueNode) or \
                    isinstance(node.arg, FalseNode):
                    return node.token + parse_helper(node.arg)
                else:
                    return node.token + "(" + parse_helper(node.arg) + ")"
            else:
                return "(" + parse_helper(node.left) + node.token + parse_helper(node.right) + ")"

        return parse_helper(self.root)





class LogicTreeTrainer():

    def __init__(self, first_tree=None):

        if first_tree != None:
            self.starting_expr = first_tree
            tree_postfix = self.inToPostFix(first_tree)
            tree = LogicTree(tree_postfix)
            self.trees = {1 : (tree, [(tree.copy(), 0)])}

        else:
            self.starting_expr = 'T'
            tree = LogicTree(self.starting_expr)
            self.trees = {1 : (tree, [(tree.copy(), 0)])}

        self.ops = 0



    def inToPostFix(self, s):
        """
        Got this function from here:

        https://stackoverflow.com/questions/57227009/
            need-an-algorithm-to-convert-propositional-logic-expression-from-infix-to-postfi

        """


        def reject(what): # Produce a readable error
            raise SyntaxError("Expected {}, but got {} at index {}".format(
                what or "EOF",
                "'{}'".format(tokens[-1]) if tokens else "EOF",
                len(s) - len(tokens)
            ))

        get = lambda: tokens.pop() if tokens else ""
        put = lambda token: output.append(token)
        match = lambda what: tokens[-1] in what if tokens else what == ""
        expect = lambda what: get() if match(what) else reject(what)

        def suffix():
            token = get()
            term()
            put(token)

        def parens():
            expect("(")
            expression(")")

        def term():
            if match(identifier): put(get())
            elif match(unary): suffix()
            elif match("("): parens()
            else: expect("an identifier, a unary operator or an opening parenthesis");

        def expression(terminator):
            term()
            if match(binary): suffix()
            expect(terminator)

        # Define the token groups
        identifier = "abcdefghijklmnopqrstuwxyz"
        identifier += identifier.upper()
        unary = "~";
        binary = "^∧v∨→";
        tokens = list(reversed(s)) # More efficient to pop from the end
        output = [] # Will be populated during the parsing
        expression("") # Parse!
        return "".join(output)


    def increment_ops(self, ops=1):

        for i in range(ops):
            new_tree_dict = copy.deepcopy(self.trees)
            max_op_num = len(new_tree_dict[len(new_tree_dict)][1])
            for (id, treetup) in self.trees.items():
                if len(treetup[1]) == max_op_num:
                    tree = treetup[0]
                    new_trees = tree.deep_ops()
                    for new_tree in new_trees:
                        new_sequence = treetup[1].copy()
                        new_sequence.append((new_tree[0], new_tree[1]))
                        new_tree_dict[max(new_tree_dict.keys()) + 1] = (new_tree[0], new_sequence)

            # print(len(new_tree_dict))
            # REMOVING DUPLICATES (WARNING, ALSO REMOVES CORRECTLY FORMED DUPLICATES)
            exprs = set()
            idx = 1
            unduplicated_new_trees = dict()
            i = 1
            for id, treetup in new_tree_dict.items():
                expr = treetup[0].parse_tree()
                if expr not in exprs:
                    exprs.add(expr)
                    unduplicated_new_trees[idx] = treetup
                    idx += 1
            new_tree_dict = unduplicated_new_trees



            self.ops += 1
            self.trees = new_tree_dict
            print(len(self.trees))


    def duplicates_info(self):

        dataset = self.get_tree_sequences()
        trees = []
        for i in range(len(dataset)):
            treetup = dataset[i]
            trees.append((treetup[0].parse_tree(), len(treetup[1]) - 1, treetup[1]))

        max_op_num = self.ops
        op_depths_dict = {i:[] for i in range(max_op_num+1)}

        for treetup in trees:
            op_depths_dict[treetup[1]].append(treetup)

        all_duplicates = []
        for depth in range(len(op_depths_dict)):

            dup_dict = dict()
            checktrees = op_depths_dict[depth]
            passed_strs = []
            for treetup in checktrees:
                seq = []
                for tree in treetup[2]:
                    seq.append(tree[0].parse_tree())

                if treetup[0] in passed_strs:
                    if treetup[0] in dup_dict.keys():
                        dup_dict[treetup[0]].append(seq)
                    else:
                        dup_dict[treetup[0]] = [seq]
                else:
                    dup_dict[treetup[0]] = [seq]
                    passed_strs.append(treetup[0])

            dup_dict2 = dict()
            for expr, occurences in dup_dict.items():
                if len(occurences) > 1:
                    dup_dict2[expr] = occurences

            all_duplicates.append(dup_dict2)

            print(depth, len(dup_dict2)/len(checktrees))


        return all_duplicates


    def cross_depth_dups_info(self, prev, cur):
        dataset = self.get_tree_sequences()
        trees = []
        for i in range(len(dataset)):
            treetup = dataset[i]
            trees.append((treetup[0].parse_tree(), len(treetup[1]) - 1, treetup[1]))

        max_op_num = self.ops
        op_depths_dict = {i:[] for i in range(max_op_num+1)}

        for treetup in trees:
            op_depths_dict[treetup[1]].append(treetup)

        depth_strs = [[] for i in range(len(op_depths_dict))]

        for i in range(len(depth_strs)):
            full_list = op_depths_dict[i]
            depth_strs[i].extend([treetup[0] for treetup in full_list])

        cur_strs = set(depth_strs[cur])
        prev_strs = set(depth_strs[prev])

        all_results_tup = []
        cur_full_lists = op_depths_dict[cur]
        prev_full_lists = op_depths_dict[prev]
        for expr in list(cur_strs.intersection(prev_strs)):
            results_tup = ([],[])
            for treetup in prev_full_lists:
                if treetup[0] == expr:
                    results_tup[0].append(treetup)
            for treetup in cur_full_lists:
                if treetup[0] == expr:
                    results_tup[1].append(treetup)

            all_results_tup.append(results_tup)

        return all_results_tup




        return list(cur_strs.intersection(prev_strs))


    def get_trees(self):
        return [tup[0] for tup in list(self.trees.values())]
    def get_sequences(self):
        return [tup[1] for tup in list(self.trees.values())]
    def get_tree_sequences(self):
        return [tup for tup in list(self.trees.values())]













if __name__ == '__main__':

    trainers = []

    # starting_exprs = ['T', 'F', 'p', 'q', 'r', '~p', '~q', '~r']
    # starting_exprs.extend(['p→q'])

    starting_exprs = ['T']
    for expr in starting_exprs:
        print("building trees from " + expr)
        trainer = LogicTreeTrainer(expr)
        trainer.increment_ops(4)
        trainers.append(trainer)


    for i in range(len(trainers)):
        trainer = trainers[i]
        seed = starting_exprs[i]
        loc = '../data/logic_proof_datasets/' + seed + '.pkl'
        pkl.dump(trainer, open(loc, 'wb'))













































# comment
