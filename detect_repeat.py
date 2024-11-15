from tree_sitter import Language, Parser
import tree_sitter_python
import collections

PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)



def has_repeating_pattern(lst, repeat_num=2):
    n = len(lst)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            pattern = lst[-i:]
            count = 0
            for j in range(n, 0, -i):
                if lst[j-i:j] == pattern:
                    count += 1
                else:
                    break
                if count >= repeat_num:
                    return True, pattern
    return False, None


def detect_repeat_pattern(code, max_repeats=2):

    result = {
        "message": "",
        "error_type": "",
        "error_message": "",
        "error_line": "",
        "error_lineno": -1,
        "error_offset": -1,
        "test_error" : False
    }    

    is_repeat, repeat_first_line = _detect_repeat_pattern(code, max_repeats)
    if is_repeat:
        result["message"] = "Repeated code detected"
        result["error_type"] = "RepeatPatternError"
        result["error_line"] = None
        result["error_lineno"] = repeat_first_line
        result["error_offset"] = 0
    else:
        result["message"] = "Code Test Passed."

    return result


def _detect_repeat_pattern(code, max_repeats=2):
    tree = parser.parse(bytes(code, "utf8"))
    repeat_first_line = 0
    last_node_type = None
    last_node_type_count = 0

    queue = collections.deque([[tree.root_node]])
    while queue:
        node_l = queue.popleft()
        for node in node_l:
            if not ('module' in node.type or 'statement' in node.type or 'comment' in node.type or 'block' in node.type or 'else_clause' in node.type or 'elif_clause' in node.type or 'function_definition' in node.type or 'class_definition' in node.type):
                continue

            node_type = node.type
            if node_type == 'expression_statement':
                node_type = node.children[0].type
            if node_type == last_node_type:
                last_node_type_count += 1
                if last_node_type_count >= max_repeats:
                    return True, repeat_first_line
            else:
                last_node_type = node_type
                last_node_type_count = 0
                repeat_first_line = node.start_point[0] + 1

            if node.children:
                queue.extend([node.children])
        
        last_node_type = None
        last_node_type_count = 0
        repeat_first_line = 0

    return False, -1