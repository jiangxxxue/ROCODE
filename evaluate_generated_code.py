import json
import argparse
import copy

from evaluate.execute.execution import evaluate_with_test_code
from evaluate.evaluation import pass_at_K, AvgPassRatio, CCP
from utils import load_dataset_my, load_dataset_map_my, find_method_name_mbpp, build_test_method, truncate_back_no_signature, build_AVG_solutions, build_CCP_solutions, build_test_method_for_CodeForces, post_process_code_for_CodeForces


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument("--truncate", action="store_true", help="If set, will truncate completion.")
parser.add_argument("--k_list", type=int, nargs='+', default=[1,2,5,10])
parser.add_argument("--eval_standard", action="store_true")
parser.add_argument("--eval_ET", action="store_true")
args = parser.parse_args()


assert args.dataset in ['MBPP', 'humaneval', 'CodeForces2305'], "Dataset not supported"

# load raw dataset
raw_dataset = load_dataset_my(args.dataset)
raw_dataset_map = load_dataset_map_my(args.dataset)
summary_f = open(args.input_path + "_results_summary", 'a+')
summary_f.write("-----------------------new evaluation-----------------------\n")

# excepted tasks can be added here
except_list = []
raw_handled_solutions = []
with open(args.input_path, 'r') as f:
    for line in f:
        line = json.loads(line)
        if line["task_id"] in except_list:
            continue
        assert "prompt" in line.keys(), "prompt not in line"
        assert "completion" in line.keys(), "completion not in line"

        if args.dataset == "MBPP":
            line["entry_point"] = find_method_name_mbpp(raw_dataset_map[line["task_id"]]['test_list'])
            line["test"] = build_test_method(raw_dataset_map[line["task_id"]]['test_list'], raw_dataset_map[line["task_id"]]['test_setup_code'], line["entry_point"])
        elif args.dataset == "humaneval":
            line["entry_point"] = raw_dataset_map[line["task_id"]]['entry_point']
            line["test"] = raw_dataset_map[line["task_id"]]['test']
        elif args.dataset == "CodeForces2305":
            line["entry_point"] = 'solution'
            input_output = raw_dataset_map[int(line["task_id"].split("/")[1])]['test']
            line["test"] =  build_test_method_for_CodeForces(input_output)
            line["completion"] = post_process_code_for_CodeForces(prompt=line['prompt'], code=line['completion'], func_name=line['entry_point'], m_indent='    ')
            line["prompt"] = ""            
        
        if args.truncate and not args.dataset == "CodeForces2305":
            line["completion"] = truncate_back_no_signature(line["completion"])

        raw_handled_solutions.append(line)


if args.eval_standard:

    handled_solutions = copy.deepcopy(raw_handled_solutions)

    # pass@k
    exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
    with open(args.input_path + "_results", 'w') as f:
        for idx, result in enumerate(exec_result):
            f.write(json.dumps(result) + '\n')
        f.flush()
    summary_f.write(json.dumps(pass_at_K(exec_result, k=[1,2,3,4,5,10])) + '\n')
        
    # AvgPassRatio
    handled_solutions_AVG = build_AVG_solutions(handled_solutions)
    exec_result_AVG = evaluate_with_test_code(handled_solutions_AVG, timeout=10)
    avg_pass_ratio = AvgPassRatio(exec_result_AVG)
    summary_f.write("AvgPassRatio: " + str(avg_pass_ratio) + '\n')

    # Compiler Correctness Percentage (CCP)
    handled_solutions_CCP = build_CCP_solutions(handled_solutions)
    exec_result_CCP = evaluate_with_test_code(handled_solutions_CCP, timeout=10)
    CCP_score = CCP(exec_result_CCP)
    summary_f.write("Compiler Correctness Percentage: " + str(CCP_score) + '\n')

    print('pass rates of solutions')
    print(pass_at_K(exec_result, k=args.k_list))

    print("AvgPassRatio: ", avg_pass_ratio)

    print('Compiler Correctness Percentage: ', CCP_score)


# --------------------------ET version for Humaneval and MBPP----------------------------------

if args.eval_ET:

    handled_solutions = copy.deepcopy(raw_handled_solutions)

    summary_f.write("---------------ET version----------------\n")
    print("---------------ET version----------------")

    # More Test Cases
    if args.dataset == 'MBPP':
        test_case_path= 'data/MBPP_ET.jsonl'
        with open(test_case_path, 'r') as f:
            test_cases = [json.loads(line) for line in f]
        
        test_cases_dict = {}
        for case in test_cases:
            test = build_test_method(case['test_list'], case['test_setup_code'], case['entry_point'])
            test_cases_dict[case['task_id']] = test

    elif args.dataset == "humaneval":
        test_case_path= 'data/HumanEval_ET.jsonl'
        with open(test_case_path, 'r') as f:
            test_cases = [json.loads(line) for line in f]
            
        test_cases_dict = {}
        for case in test_cases:
            test = build_test_method(case['test_case_list'], "", case['entry_point'])
            test_cases_dict[case['task_id']] = test


    for solution in handled_solutions:
        solution['test'] =test_cases_dict[solution['task_id']]

    # pass@k
    exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
    summary_f.write(json.dumps(pass_at_K(exec_result, k=[1,2,3,4,5,10])) + '\n')
    
    # AvgPassRatio
    handled_solutions_AVG = build_AVG_solutions(handled_solutions)
    exec_result_AVG = evaluate_with_test_code(handled_solutions_AVG, timeout=10)
    avg_pass_ratio = AvgPassRatio(exec_result_AVG)
    summary_f.write("AvgPassRatio: " + str(avg_pass_ratio) + '\n')

    print('pass rates of solutions')
    print(pass_at_K(exec_result, k=args.k_list))
    print("AvgPassRatio: ", avg_pass_ratio)

summary_f.close()