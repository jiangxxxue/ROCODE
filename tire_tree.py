class TreeNode:
    def __init__(self, token, token_id, prob=0, token_entropy=0, decay=1.0):
        self.node_id = 0
        self.token = token
        self.token_id = token_id
        self.lineno = 0
        self.prob = prob
        self.token_entropy = token_entropy
        self.decay = decay
        self.children = {}
        self.parent = None


    def add_child(self, raw_token, lineno, max_node_id):
        for key, value in raw_token.items():
            token = key
            prob, token_id, token_entropy  = value
            
        if token_id not in self.children.keys():
            self.children[token_id] = TreeNode(token, token_id, prob, token_entropy)
            self.children[token_id].parent = self
            self.children[token_id].lineno = lineno
            self.children[token_id].node_id = max_node_id + 1
        else:
            self.children[token_id].prob = prob

        return self.children[token_id]


class TireTree:
    def __init__(self, code_context):
        self.root = TreeNode(token = "root", token_id = -1)
        self.root.lineno = len(code_context.rstrip().split("\n"))
        self.rollback_node = self.root
        self.current_legal_node = self.root
        self.node_num = 0
        self.token_budget = 0
        self.dfc = 0.5


    def find_line_start_node(self, lineno):
        # RollBack() guarantees that lineno must be rooted to the current_legal_node
        node = self.current_legal_node
        router = []
        while node.parent:
            router.append(node.node_id)
            if node.lineno == lineno and node.parent.lineno == lineno-1:
                return node, router
            node = node.parent

        # Jumping out of the while loop indicates that lineno is the line number of root
        assert node.token == "root"
        return node, router


    def update(self, new_code):
        current_node = self.current_legal_node
        lineno = current_node.lineno + 1
        for idx, token in enumerate(new_code):
            current_node = current_node.add_child(token, lineno, self.node_num)
            if current_node.node_id > self.node_num:
                self.node_num = current_node.node_id
        self.current_legal_node = current_node


    def decay_path(self, rollback_point, decay_factor=0.9):
        self.rollback_node, router = self.find_line_start_node(rollback_point[0])
        rollback_length = len(router)

        assert self.rollback_node is not None, f"rollback_node is None, rollback_point: {rollback_point}"

        if self.rollback_node.lineno == self.root.lineno:
            pass
        else:
            self.rollback_node = self.rollback_node.parent  

        node = self.rollback_node
        for i, node_id in enumerate(router[::-1]):
            for _, child in node.children.items():
                if child.node_id == node_id:
                    child.decay = child.decay * (decay_factor ** i)
                    if child.token == '#':
                        child.decay = child.decay * self.dfc
                    node = child
                    break
        self.current_legal_node = self.rollback_node
        return rollback_length


    def get_legal_code(self, dataset, tokenizer):
        current_node = self.current_legal_node
        legal_code = []
        while current_node != self.root:
            legal_code.append(current_node.token_id)
            current_node = current_node.parent
        
        if len(legal_code) == 0:
            return ""

        if dataset == "MBPP":
            return tokenizer.decode(legal_code[::-1], skip_special_tokens=True)
        
        if "llama" in tokenizer.name_or_path.lower():
            return " " + tokenizer.decode(legal_code[::-1], skip_special_tokens=True)
        
        return tokenizer.decode(legal_code[::-1], skip_special_tokens=True)
    

    def get_legal_code_ids(self):
        current_node = self.current_legal_node
        legal_code = []
        while current_node != self.root:
            legal_code.append(current_node.token_id)
            current_node = current_node.parent
            
        return legal_code[::-1]


    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print(f"token: {node.token:<12}, lineno: {node.lineno:<8}, token_entropy: {node.token_entropy:<10.3f}, decay: {node.decay:<10.3f}")
        for child in node.children.values():
            self.print_tree(child, level + 1)
    

    def find_node(self, target_node):
        return self.find_node_recursive(self.root, target_node)
    

    def find_node_recursive(self, node, target_node):
        if node.action == target_node:
            return node
        
        for child in node.children.values():
            result = self.find_node_recursive(child, target_node)
            if result:
                return result
        
        return None