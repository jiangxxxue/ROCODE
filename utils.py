import re
import json
import ast
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk


def process_indented_block(code):
    if code.rstrip().endswith(":"):
        last_line = code.rstrip("\n").split("\n")[-1]
        if last_line.lstrip().startswith("while"):
            code = code.rstrip("\n")+" break \n"
        else:
            code = code.rstrip("\n")+" pass \n"
    return code


def add_break_if_in_while_block(code):
    tree = ast.parse(code)
    last_line_number = len(code.rstrip().split('\n'))
    
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            
            while_start_line = node.lineno
            while_end_line = node.end_lineno
            
            if while_start_line <= last_line_number <= while_end_line and not any(isinstance(n, ast.Break) for n in node.body):
                node.body.append(ast.Break())
                break
    
    modified_code = astor.to_source(tree)
    return modified_code
    

from tree_sitter import Language, Parser
import tree_sitter_python
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)
def find_node_by_line(node, line_number):
    if node.start_point[0] <= line_number <= node.end_point[0]:
        if node.child_count == 0:
            return node
        for child in node.children:
            result = find_node_by_line(child, line_number)
            if result:
                return result
    return None


def is_inside_loop(dataset, code):
    if dataset=="MBPP" or dataset=="humaneval":
        tree = parser.parse(bytes(code, "utf8"))
        lines = code.rstrip().split('\n')
        last_line_number = len(lines)-1
        node = find_node_by_line(tree.root_node, last_line_number)
        while node:
            if node.type in ["for_statement", "while_statement"]:
                return True
            node = node.parent
        return False

    else:
        pass

    return False


def is_complete(dataset, requirement, completion, num_newlines=3):

    if dataset in ["MBPP", "humaneval", "CodeForces2305"]:
        requirement_tree = parser.parse(bytes(requirement, "utf8"))
        tree = parser.parse(bytes(requirement + completion, "utf8"))
        if tree.root_node.type == "ERROR":
            return False
        if len(tree.root_node.children) > len(requirement_tree.root_node.children):
            return True
    else:
        pass

    if completion.endswith('\n' * (num_newlines + 1)):
        return True
        
    return False
        


import math

def calculate_positional_entropy(output_distribution):
    position_entropy = 0
    output_distribution = output_distribution.tolist()
    for prob in output_distribution:
        if prob > 0:
            position_entropy += -prob * math.log2(prob)
    return position_entropy


def calculate_entropy_deviation(entropies):
    average_entropy = sum(entropies) / len(entropies)
    deviations = [(entropy - average_entropy) for entropy in entropies]
    return deviations, average_entropy


def calculate_variance(entropies):
    average_entropy = sum(entropies) / len(entropies)
    variance = sum((entropy - average_entropy) ** 2 for entropy in entropies) / len(entropies)
    return variance, average_entropy




def prompt_split_humaneval(prompt, mehotd_name):
    prompt = prompt.strip()
    prompt = prompt.replace("\r\n", "\n")
    before_func = prompt[:prompt.rfind("def ")]
    code = prompt[prompt.rfind("def "):]

    comment_start_1 = re.search("\"\"\"", code)
    comment_start_2 = re.search("\'\'\'", code)
    if comment_start_1:
        comment_start = comment_start_1.end()
    elif comment_start_2:
        comment_start = comment_start_2.end()

    example_start_1 = re.search("[eE]xample(:)?", code)
    example_start_2 = re.search("[fF]or [eE]xamble(:)?", code)
    example_start_3 = re.search(">>>", code)
    example_start_4 = re.search(mehotd_name+"\(.+\)", code[comment_start:])

    if example_start_1:
        comment = code[comment_start:example_start_1.start()]
        example = code[example_start_1.start():-4]
    elif example_start_2:
        comment = code[comment_start:example_start_2.start()]
        example = code[example_start_2.start():-4]
    elif example_start_3:
        comment = code[comment_start:example_start_3.start()]
        example = "Example:\n"+code[example_start_3.start():-4]
    elif example_start_4:
        comment = code[comment_start:example_start_4.start()+comment_start]
        example = "Example:\n"+code[example_start_4.start()+comment_start:-4]
    else:
        comment = code[comment_start:-4]
        example = ""
    comment = comment.strip().replace("\n", " ")
    comment = re.sub("\s+", " ", comment)

    example = re.sub("\n(\s)*","\n\t",example)
    test_case = "\t"+example.strip()
    signature = code[:code.index("\n")+1]

    return before_func, signature, comment, test_case



def __add_line_numbers(code):
    lines = []
    for lineno, line in enumerate(code.split('\n'), 1):
        lines.append(f'line {lineno}: {line}')
    return "\n".join(lines)


def truncate_back(d, method_name=""):
    pred = d[:d.find('def '+method_name)]
    d = d[d.find('def '+method_name):]
    line = d.split('\n')

    code = [line[0]]
    for l in line[1:]:
        if len(l.strip()) == 0:
            code.append(l)
            continue
        indent = len(l) - len(l.lstrip())
        if indent == 0:
            break
        else:
            code.append(l)

    return pred + '\n'.join(code).strip()


def truncate_back_no_signature(d):
    line = d.split('\n')
    code = []
    for l in line:
        if len(l.strip()) == 0:
            code.append(l)
            continue
        indent = len(l) - len(l.lstrip())
        if indent == 0:
            break
        else:
            code.append(l)

    return '\n'.join(code)


def load_dataset_my(dataset_name):
    if dataset_name == "MBPP":
        dataset =load_dataset("mbpp")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "humaneval":
        dataset =load_dataset("openai/openai_humaneval")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "CodeForces2305":
        dataset = load_from_disk("data/CodeForces2305")            
    else:
        raise ValueError("dataset_name not found")
    return dataset


def load_dataset_map_my(dataset_name):
    if dataset_name == "MBPP":
        dataset =load_dataset("mbpp")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "humaneval":
        dataset =load_dataset("openai/openai_humaneval")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "CodeForces2305":
        dataset = load_from_disk("data/CodeForces2305")
    
    dataset_map = {}
    for i in tqdm(range(len(dataset))):
        dataset_map[dataset[i]["task_id"]] = dataset[i]
    return dataset_map


def format_prompt(task_id, text, tests, sample_code, num_use_cases=0):
    # Create prompt from scratch
    prompt = f'\t"""\n\t{text}\n\n'
    if num_use_cases > 0:
        for i in range(num_use_cases):
            example = tests[i].split("assert ")[-1].replace("==", "=")
            prompt += f"\t>>> Example: {example}\n"

    # Add code prefix
    fn_name = tests[0].split("assert ")[-1]

    # special case for MBPP
    if fn_name.startswith("(") or fn_name.startswith("int("):
        fn_name = fn_name.split("(")[1]
    else:
        fn_name = fn_name.split("(")[0]
    fn_name = fn_name.strip()
    fn_search = re.search(f"def {fn_name}\s?\(.*\)\s?:", sample_code)

    if fn_search is None:
        raise ValueError(
            f"Could not find 'def {fn_name}\(.*\):' in code for task {task_id}."
        )
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'{code_prefix}\n{prompt}\t"""\n'
    return prompt, fn_name


def find_method_name_mbpp(tests):
    fn_name = tests[0].split("assert ")[-1].split("(")[0]
    fn_name = fn_name.strip()
    return fn_name


# Restrictions: The syntax of the program should be correct
def get_function_name_body(code):

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.function_bodies = {}

        def visit_FunctionDef(self, node):
            start_lineno = node.body[0].lineno - 1
            if hasattr(node.body[-1], "end_lineno"):
                end_lineno = node.body[-1].end_lineno
            else:
                end_lineno = node.body[-1].lineno
            function_body = "\n".join(code.splitlines()[start_lineno:end_lineno])

            self.function_bodies[node.name] = function_body

    parsed_code = ast.parse(code)
    visitor = FunctionVisitor()
    visitor.visit(parsed_code)

    for func_name, body in visitor.function_bodies.items():
        function_name = func_name
        function_body = body

    return function_name, function_body


def save_to_file(data, file_path):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def build_test_method(test_list, test_imports, method_name):
    if test_imports:
        test_imports = "\n".join(test_imports)
        test_method = test_imports + "\n"
    else:
        test_method = ""
    test_method = "def check(" + method_name + "):\n"
    if len(test_list) == 0:
        return test_method + "\treturn True" + "\n"
    for test in test_list:
        test_method += '\t' + test + "\n"
    return test_method.strip("\n")


def build_test_method_for_CodeForces(input_output):
    test_method = "def check(candidate):\n"
    for idx, (input, output) in enumerate(zip(input_output['inputs'], input_output['outputs'])):
        try:
            test_method += "\tassert candidate(%r) == %r \n" % (input.strip(), output.strip())
        except:
            test_method += "\tassert candidate(%s) == %s \n" % (input, output)
    return test_method


import json
import os
import re
import pdb


def truncate(d):
    d = d.split('\n\n')
    s = d[0] + '\n\n'
    if len(d)>1:
        for i in d[1:]:
            if 'def' not in i and i and '__main__' not in i:
                s += i + '\n\n'
            else:
                break
    return s


def minimum_indent(lines):
    m_indent = 100
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if indent > 0 and indent < m_indent:
            m_indent = indent
    return m_indent


def check_overall_indent(lines):
    def check_indent(lines):
        for line in lines:
            if "def" not in line and "print" not in line and "__name__" not in line and line[0] != '#' and len(line) - len(line.lstrip()) == 0:
                return True
        return False
    m_indent = minimum_indent(lines)
    if len(lines) <= 1:
        return False
    elif len(lines[0]) - len(lines[0].lstrip()) == 0:
        if lines[0].strip()[-1] == ':':
            space_num = len(lines[1]) - len(lines[1].lstrip())
            if space_num == m_indent:
                return True
        elif check_indent(lines[1:]):
            return True
    return False


def post_process_code_for_CodeForces(prompt, code, func_name, m_indent):
    assert type(code) == str
    if f"def {func_name}(" in code:
        return code
    truncation = truncate(code).replace('\r', '\n')
    truncation = re.sub('\n+', '\n', truncation)
    lines = truncation.split('\n')
    lines = list(filter(lambda x: x.strip() != "", lines))
    lines = list(map(lambda x: x.replace('\t', m_indent), lines))

    if len(lines) == 0:
        pass
    else:
        if check_overall_indent(lines):
            for i in range(len(lines)):
                lines[i] = m_indent + lines[i] 
        elif len(lines[0]) - len(lines[0].lstrip()) == 0:
            lines[0] = m_indent + lines[0]
        else:
            pass
    return prompt.replace('\t', m_indent)+'\n'.join(lines)




import copy
import astor

class FunctionBodyFinder(ast.NodeVisitor):
    def __init__(self, target_func_name):
        self.target_func_name = target_func_name
        self.function_body_node = None

    def visit_FunctionDef(self, node):
        if node.name == self.target_func_name:
            self.function_body_node = node.body

class FunctionBodyModifier(ast.NodeTransformer):
    def __init__(self, target_func_name, new_node):
        self.target_func_name = target_func_name
        self.new_node = new_node

    def visit_FunctionDef(self, node):
        if node.name == self.target_func_name:
            node.body = self.new_node
        return node




def build_AVG_solutions(solutions):
    AVG_solutions = []
    finder = FunctionBodyFinder(target_func_name= "check")
    
    for sidx, solution in enumerate(solutions):
        solution["task_id"] = str(solution["task_id"]) + "_" + str(sidx)
        test = solution["test"]
        try:
            ast_tree = ast.parse(test)
        except:
            print("ast parse error")
            continue
        finder.visit(ast_tree)
        body_statement = copy.deepcopy(finder.function_body_node)
        assert_statement = [statement for statement in body_statement if type(statement) == ast.Assert]
        body_statement = [statement for statement in body_statement if type(statement) != ast.Assert]
        if len(assert_statement) == 0:
            AVG_solutions.append(copy.deepcopy(solution))
        
        for idx, statement in enumerate(assert_statement):
            assert type(statement) == ast.Assert
            modifier = FunctionBodyModifier(target_func_name= "check", new_node=body_statement + [statement])
            modified_test = modifier.visit(ast_tree)
            AVG_solution = copy.deepcopy(solution)
            AVG_solution["task_id"] = str(solution["task_id"]) + "_AVG_" + str(idx)
            AVG_solution["test"] = astor.to_source(modified_test)
            AVG_solutions.append(AVG_solution)
    return AVG_solutions


def get_test_input_from_assert(assert_stmt):
    if isinstance(assert_stmt.test, ast.Compare):
        left_expr = assert_stmt.test.left
        if isinstance(left_expr, ast.Call) and isinstance(left_expr.func, ast.Name):
            func_name = left_expr.func.id
            args_str = ', '.join(ast.unparse(arg) for arg in left_expr.args)
            function_call = f"{func_name}({args_str})"
            return function_call
    elif isinstance(assert_stmt.test, ast.Call):
        return ast.unparse(assert_stmt.test)
    return ""


def extract_code_from_location(source_code, node):
    lines = source_code.splitlines()
    
    start_line = node.lineno - 1
    start_col = node.col_offset
    
    if hasattr(node, 'end_lineno') and hasattr(node, 'end_col_offset'):
        end_line = node.end_lineno - 1
        end_col = node.end_col_offset
        if start_line == end_line:
            return lines[start_line][start_col:end_col]
        else:
            code_lines = [lines[start_line][start_col:]]
            code_lines.extend(lines[start_line + 1:end_line])
            code_lines.append(lines[end_line][:end_col])
            return "\n".join(code_lines)
    else:
        return lines[start_line][start_col:]


def build_CCP_solutions_old(solutions):
    CCP_solutions = []
    finder = FunctionBodyFinder(target_func_name= "check")
    
    for sidx, solution in enumerate(solutions):
        solution["task_id"] = str(solution["task_id"]) + "_" + str(sidx)
        test = solution["test"]
        try:
            ast_tree = ast.parse(test)
        except:
            print("ast parse error")
            continue
        finder.visit(ast_tree)
        body_statement = copy.deepcopy(finder.function_body_node)
        assert_statement = [statement for statement in body_statement if type(statement) == ast.Assert]

        if len(assert_statement) == 0:
            CCP_solutions.append(copy.deepcopy(solution))
        
        CCP_test = test
        for idx, statement in enumerate(assert_statement):
            assert type(statement) == ast.Assert
            CCP_test = CCP_test.replace(extract_code_from_location(test,statement), get_test_input_from_assert(statement))

        CCP_solution = copy.deepcopy(solution)
        CCP_solution["task_id"] = str(solution["task_id"]) + "_CCP_" + str(idx)
        CCP_solution["test"] = CCP_test
        CCP_solutions.append(CCP_solution)

    return CCP_solutions


def build_CCP_solutions(solutions):
    CCP_solutions = []
    finder = FunctionBodyFinder(target_func_name= "check")
    
    for sidx, solution in enumerate(solutions):
        test = solution["test"]
        try:
            ast_tree = ast.parse(test)
        except:
            print("ast parse error")
            continue
        finder.visit(ast_tree)
        body_statement = copy.deepcopy(finder.function_body_node)
        assert_statement = [statement for statement in body_statement if type(statement) == ast.Assert]

        CCP_test = test
        for idx, statement in enumerate(assert_statement):
            assert type(statement) == ast.Assert
            CCP_test = CCP_test.replace(extract_code_from_location(test,statement), get_test_input_from_assert(statement))

        CCP_solution = copy.deepcopy(solution)
        CCP_solution["task_id"] = str(solution["task_id"]) + "_CCP_" + str(idx)
        CCP_solution["test"] = CCP_test
        if len(assert_statement) == 0:
            CCP_solution["test"] = 'def check(candidate):' + '\n' + '\t' + 'return True'

        CCP_solutions.append(CCP_solution)

    return CCP_solutions
    