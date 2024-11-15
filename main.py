import os
import argparse
import json
import logging
import pprint
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from PA_tools import PA
from tire_tree import TireTree
from detect_repeat import detect_repeat_pattern
from utils import load_dataset_my, format_prompt, calculate_positional_entropy, is_complete, process_indented_block, add_break_if_in_while_block
from config import MAX_GENERATION_LENGTH, MAX_ATTEMPT_NUM, TOKEN_BUDGET

import warnings
warnings.filterwarnings('ignore')




def top_k_p_sample(probs, top_k, top_p):
    sorted_indices = torch.argsort(probs, descending=True)[0]
    sorted_probs = probs[0][sorted_indices]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    last_index = torch.where(cumulative_probs >= top_p)[0]
    last_index = last_index[0] if len(last_index)!=0 else len(cumulative_probs)
    sorted_indices = sorted_indices[:last_index+1]
    sorted_indices = sorted_indices[:top_k]
    final_probs = probs[0][sorted_indices]
    final_probs /= final_probs.sum()
    return final_probs, sorted_indices


def generate_code_stat(args, requirement, generated_code_ids, tire_tree, model, tokenizer, max_generation_length):
    completions = []
    completions_probs = []

    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(requirement, add_special_tokens=False, verbose=False) 
    input_ids = torch.tensor([input_ids + generated_code_ids]).to("cuda:0")

    model.eval()
    try:
        with torch.no_grad():
            completion = []
            cur_node = tire_tree.current_legal_node
            decay_flag = True
            for i in range(max_generation_length):
                tokens = model.generate(
                    input_ids,
                    num_return_sequences=1,
                    max_length=input_ids.shape[1]+1,
                    use_cache=True,
                    do_sample=False,
                    return_dict_in_generate=True, 
                    output_scores=True,
                )
                scores = tokens.scores[0]
                if args.temperature > 0:
                    scores = scores / args.temperature
                    do_sample = True                        
                else:
                    do_sample = False
                probs = torch.softmax(scores, dim=-1)
                token_entropy = calculate_positional_entropy(probs[0])

                # Decaying the probability of some token
                if cur_node.children and decay_flag:
                    for t in cur_node.children.keys():
                        probs[0][t] = probs[0][t] * cur_node.children[t].decay
                
                if do_sample == False:
                    generated_token_prob, generated_token_prob_index = probs[0].max().item(), probs[0].argmax().item()
                else:
                    final_probs, sorted_indices = top_k_p_sample(probs, args.topk, args.topp)
                    generated_token_prob_index = sorted_indices[torch.multinomial(final_probs, num_samples=1).item()].item()
                    generated_token_prob = probs[0][generated_token_prob_index].item()
                
                completions_probs.append({tokenizer.decode(generated_token_prob_index):(generated_token_prob, generated_token_prob_index, token_entropy)})
                completion.append(generated_token_prob_index)
                if generated_token_prob_index != tokens.sequences[0][-1]:
                    tokens.sequences[0][-1] = generated_token_prob_index
                input_ids = tokens.sequences[0].unsqueeze(0)

                # Update the current node
                if (generated_token_prob_index in cur_node.children) and decay_flag:
                    cur_node = cur_node.children[generated_token_prob_index]
                else:
                    decay_flag = False

                # End of statement, stop generating
                if "\n" in tokenizer.decode(generated_token_prob_index):
                    break

            completions.append(tokenizer.decode(completion, skip_special_tokens=True))

    except RuntimeError as e:
        logging.error(f"Could not sample from model: {e}")

    return completions, completions_probs


def tokenizer_decode(dataset, token_ids, tokenizer):
    if dataset == "MBPP":
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    if "llama" in tokenizer.name_or_path.lower():
        return " " + tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def load_model_tokenizer(args, arch, model_dir):
    if model_dir:
        model_path = model_dir
    elif "codegen" in arch:
        model_path = f"Salesforce/{args.arch}"
    elif "Llama" in arch:
        model_path = f"meta-llama/{args.arch}"
    else:
        raise ValueError(f"Unknown model architecture: {arch}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def PA_tool(tire_tree, requirement, test_cases, tokenizer):
    requirement_ids = tokenizer.encode(requirement, add_special_tokens=False, verbose=False)
    raw_test_code_ids = tire_tree.get_legal_code_ids()
    raw_test_code = tokenizer.decode(requirement_ids + raw_test_code_ids, skip_special_tokens=True)

    test_code = process_indented_block(raw_test_code)         
    if "while" in "\n".join(test_code.rstrip("\n").split("\n")[:-1]):
        try:
            test_code = add_break_if_in_while_block(test_code)
        except:
            pass

    report = PA(test_code, test_cases)

    # the statement hasn't completed generating
    if "EOF" in report["message"]:
        report["message"] = "Code Test Passed."
        return report
    elif report["message"] == "Code Test Passed.":
        report = detect_repeat_pattern(test_code, max_repeats=5)

    return report


def RollBack(report, tire_tree, rollback_point, successive_error_attempt, tokenizer):
    if tire_tree.root.lineno >= report["error_lineno"] or report["error_lineno"] > tire_tree.current_legal_node.lineno+1:
        cur_node = tire_tree.current_legal_node
        max_entropy_node = cur_node
        while cur_node.parent:
            if cur_node.token_entropy > max_entropy_node.token_entropy:
                max_entropy_node = cur_node
            cur_node = cur_node.parent
        rollback_point = [max_entropy_node.lineno, 0]
        return rollback_point

    if (report["error_type"] == "SyntaxError") or ("<string>" in report["error_message"]):
        rollback_point = [report["error_lineno"], report["error_offset"]-1]
        if successive_error_attempt > MAX_ATTEMPT_NUM:
            rollback_point = [tire_tree.root.lineno, 0]
    elif report["error_type"] == "RepeatPatternError":
        rollback_point = [report["error_lineno"], 0] 
    else:
        if successive_error_attempt > MAX_ATTEMPT_NUM:
            cur_node, _ = tire_tree.find_line_start_node(report["error_lineno"])
            max_entropy_node = cur_node
            while cur_node.parent:
                if cur_node.token_entropy > max_entropy_node.token_entropy:
                    max_entropy_node = cur_node
                cur_node = cur_node.parent
            rollback_point = [max_entropy_node.lineno, 0]
        else:
            rollback_point = [report["error_lineno"], 0]

    return rollback_point


def pipeline(args, dataset, except_tasks, output_filepath, public_test_cases_map=None):

    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(args, args.arch, args.model_dir)

    # open output file
    f = open(output_filepath, "a")

    for i in tqdm(range(len(dataset))):
        task_id = dataset["task_id"][i]

        if (task_id in except_tasks):
            continue

        if args.dataset == "MBPP":
            requirement, method_name = format_prompt(
                task_id,
                dataset["text"][i],
                dataset["test_list"][i],
                dataset["code"][i],
            )
            requirement = requirement.rstrip() + "\n"
        elif args.dataset == "humaneval":
            requirement = dataset["prompt"][i]
            requirement = requirement.rstrip() + "\n"
            method_name = dataset["entry_point"][i]

        test_inputs = public_test_cases_map[task_id][0]
        full_test = public_test_cases_map[task_id][1]

        tire_tree = TireTree(requirement)
        for sample in range(args.num_samples):
            tire_tree.current_legal_node = tire_tree.root
            tire_tree.rollback_node = tire_tree.root

            generated_code_ids = []
            max_generation_length = MAX_GENERATION_LENGTH
            tire_tree.token_budget = TOKEN_BUDGET
            rollback_point = None
            successive_error_attempt = 0
            prevous_error_lineno = -1
            prevous_code = ""
            while (tire_tree.token_budget > 0) and (max_generation_length > 0):
            
                statement = generate_code_stat(args, requirement, generated_code_ids, tire_tree, model, tokenizer, max_generation_length)
                max_generation_length -= len(statement[1])
                tire_tree.token_budget -= len(statement[1])       
                statement_tokens = []
                for d in statement[1]:
                    for key, value in d.items():
                        statement_tokens.append(value[1])

                # Function-level code generation can be terminated or testing fully at the end of a function
                if (tokenizer.eos_token_id in statement_tokens) or is_complete(args.dataset, requirement, tokenizer_decode(args.dataset, generated_code_ids+statement_tokens, tokenizer)):
                    prevous_code = tire_tree.get_legal_code(args.dataset, tokenizer)
                    is_task_complete = True
                    if full_test:
                        test_cases = full_test
                    else:
                        break
                    max_generation_length += len(statement[1])
                else:
                    is_task_complete = False
                    test_cases = test_inputs
                    tire_tree.update(statement[1])

                    # If a blank line is generated or the line ends with a backslash, the test can be skipped
                    if statement[0][0].strip() == "pass" or statement[0][0].strip() == "" or statement[0][0].rstrip().endswith("\\") or statement[0][0].rstrip().endswith(","):
                        generated_code_ids = tire_tree.get_legal_code_ids()
                        continue
  
                report = PA_tool(tire_tree, requirement, test_cases, tokenizer)
                
                if report["message"] == "Code Test Passed." and is_task_complete:
                    break

                if (report["message"] != "Code Test Passed."):
                    if prevous_error_lineno == report["error_lineno"]:
                        successive_error_attempt += 1
                    else:
                        prevous_error_lineno = report["error_lineno"]
                        successive_error_attempt = 0

                    rollback_point = RollBack(report, tire_tree, rollback_point, successive_error_attempt, tokenizer)
                    rollback_length = tire_tree.decay_path(rollback_point, decay_factor=args.decay_factor)
                    max_generation_length += rollback_length

                else:
                    successive_error_attempt = 0

                generated_code_ids = tire_tree.get_legal_code_ids()
            
            if is_task_complete:
                completion = tire_tree.get_legal_code(args.dataset, tokenizer)
                completion = process_indented_block(completion)
            else:
                completion = prevous_code

            f.write(json.dumps({
                "task_id": task_id,
                "prompt": requirement,
                "completion": completion,
            }) + "\n")
            f.flush()

    f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a LLM to generate code for the benchmark.")

    parser.add_argument("--arch", default="CodeLlama-7b-hf")
    parser.add_argument("--model-dir", default=None, help="Directory where LLM checkpoints are saved.")

    parser.add_argument("--dataset", default="humaneval", type=str)
    parser.add_argument("--use_public_test_cases", default=True, action="store_true")
    parser.add_argument("--test_data", type=str, default="data/humaneval_public_test_case.jsonl")

    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--acctual-num-samples", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--topp", default=None, type=float)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--decay-factor", default=0.9, type=float)

    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    args = parser.parse_args()

    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filepath = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.arch}_temp{args.temperature}_topp{args.topp}_topk{args.topk}_df{args.decay_factor}_samples{args.num_samples}_{args.output_file_suffix}.jsonl",
    )  
        
    except_tasks = []
    if os.path.exists(output_filepath):
        print(f"File {output_filepath} already exists in {args.output_dir}.")
        lines = open(output_filepath).readlines()
        for line in lines:
            task_id = json.loads(line)["task_id"]
            if task_id not in except_tasks:
                except_tasks.append(task_id)

    dataset = load_dataset_my(args.dataset)

    public_test_cases_map = None
    if args.dataset == "humaneval":
        with open(args.test_data, "r") as f:
            test_input = [json.loads(line) for line in f]        
            if args.use_public_test_cases:
                public_test_cases_map = {d['task_id']: (d["test_input"], d["public_test_cases"]) for d in test_input}
            else:
                public_test_cases_map = {d['task_id']: (d["test_input"], None) for d in test_input}
    elif args.dataset == "MBPP":
        if args.use_public_test_cases:
            public_test_cases_map = {d['task_id']: (d["test_list"][-1].split("assert ")[-1].split("==")[0].strip(), d["test_list"][1]) for d in dataset}
        else:
            public_test_cases_map = {d['task_id']: (d["test_list"][-1].split("assert ")[-1].split("==")[0].strip(), None) for d in dataset}

    pipeline(args, dataset, except_tasks, output_filepath, public_test_cases_map)


if __name__ == "__main__":
    main(parse_args())
