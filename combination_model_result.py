# coding=utf-8
# Copyright 2022 iFLYTEK, The State Key Laboratory of Cognitive Intelligence. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# File: combination_model_result.py
# Description: combine different output line-by-line files to one line-by-line file
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/20
# Usage:


import re
import json
import argparse

def read_line_by_line_file(input_file):
    lines = []
    with open(input_file, encoding="utf-8") as f:
        for line in f.read().splitlines():
            lines.append(line)
    return lines


def convert_line_list_to_task_list(lines):
    '''
    Inputs:
        lines = [
            'User : Do you have any plain jeans? => Belief State : REQUEST:GETREQUEST:GET type = jeans, pattern = plain ] ()  <  > |  | <EOB> What do you think of the grey ones? <EOS>',
            ...
        ]
    Outputs:
        task_split_lines = [
            {
                'context': 'User : Do you have any plain jeans? => Belief State : ',
                'task1-mm-disam': '|  |',
                'task2-mm-coref': '<  >',
                'task3-mm-dst': 'REQUEST:GETREQUEST:GET type = jeans, pattern = plain ] ()',
                'task4-response': '<EOB> What do you think of the grey ones? <EOS>'
            },
            ...
        ]


    '''
    task_split_lines = []
    for line in lines:
        dst_start = line.index('Belief State : ') + 15 # 不包括'Belief State : '
        dst_end = line.index('<EOB>')
        #
        context = line[:dst_start]
        # 任务1结果
        disamb_of_dst = re.search(r"\| .*? \|",line).group() # '|  |'
        # 任务2结果
        coref_of_dst = re.search(r"< .*? >",line).group() # '<  >'
        # 任务3结果
        dst = line[dst_start:dst_end]
        dst = re.sub(r"<((<[0-9]+>)|,| )*>", "", dst)
        dst = re.sub(r"\|((<[0-9]+>)|,| )*\|", "", dst)
        right_parenthesis_index_from_right = dst.rfind(')')
        dst = dst[:right_parenthesis_index_from_right+1] # 保证dst[-1]==")", 去掉多余的空格
        # 任务4结果
        response = line[dst_end:]

        task_split_lines.append(
            {
                'context': context,
                'task1-mm-disam': disamb_of_dst,
                'task2-mm-coref': coref_of_dst,
                'task3-mm-dst': dst,
                'task4-response': response
            }
        )

    return task_split_lines


def convert_task_list_to_line_list(task_split_lines):
    '''
    Inputs:
        task_split_lines = [
            {
                'context': 'User : Do you have any plain jeans? => Belief State : ',
                'task1-mm-disam': '|  |',
                'task2-mm-coref': '<  >',
                'task3-mm-dst': 'REQUEST:GETREQUEST:GET type = jeans, pattern = plain ] ()',
                'task4-response': '<EOB> What do you think of the grey ones? <EOS>'
            },
            ...
        ]
    Outputs:
        lines = [
            'REQUEST:GETREQUEST:GET type = jeans, pattern = plain ] ()  <  > |  | <EOB> What do you think of the grey ones? <EOS>',
            ...
        ]
    '''
    lines = []
    for task_split in task_split_lines:
        line = task_split['context'] + task_split['task3-mm-dst'] + ' ' + task_split['task2-mm-coref'] + ' ' + task_split['task1-mm-disam'] + ' ' + task_split['task4-response']
        lines.append(line)
    return lines


def combination_model_result_to_one_file(
    input_line_by_line_file_for_task1=None,
    input_line_by_line_file_for_task2=None,
    input_line_by_line_file_for_task3=None,
    input_line_by_line_file_for_task4=None
    ):
    '''
    all the input files have the follow format
        User : Do you have any plain jeans? => Belief State : REQUEST:GETREQUEST:GET type = jeans, pattern = plain ] ()  <  > |  | <EOB> What do you think of the grey ones? <EOS>
        ...

    '''
    lines1 = read_line_by_line_file(input_line_by_line_file_for_task1)
    lines2 = read_line_by_line_file(input_line_by_line_file_for_task2)
    lines3 = read_line_by_line_file(input_line_by_line_file_for_task3)
    lines4 = read_line_by_line_file(input_line_by_line_file_for_task4)

    task_split_lines1 = convert_line_list_to_task_list(lines1)
    task_split_lines2 = convert_line_list_to_task_list(lines2)
    task_split_lines3 = convert_line_list_to_task_list(lines3)
    task_split_lines4 = convert_line_list_to_task_list(lines4)

    combine_split_lines = []

    for i in range(len(task_split_lines1)):
        combine_split_lines.append(
            {
                'context': task_split_lines1[i]['context'],
                'task1-mm-disam': task_split_lines1[i]['task1-mm-disam'],
                'task2-mm-coref': task_split_lines2[i]['task2-mm-coref'],
                'task3-mm-dst': task_split_lines3[i]['task3-mm-dst'],
                'task4-response': task_split_lines4[i]['task4-response']
            }
        )

    combine_lines = convert_task_list_to_line_list(combine_split_lines)

    return combine_lines, combine_split_lines


def combination_task2_result_to_one_file(
    input_line_by_line_file_for_task2_best_in_mentioned_object=None,
    input_line_by_line_file_for_task2_best_in_not_mentioned_object=None,
    mentioned_object_json_file=None):
    '''
    根据是否在mentioned_object当中选择不同的模型结果作为最终结果

    '''

    object_regex = re.compile(r"([A-Za-z0-9]+)")

    lines1 = read_line_by_line_file(input_line_by_line_file_for_task2_best_in_mentioned_object)
    lines2 = read_line_by_line_file(input_line_by_line_file_for_task2_best_in_not_mentioned_object)
    task_split_lines1 = convert_line_list_to_task_list(lines1)
    task_split_lines2 = convert_line_list_to_task_list(lines2)

    with open(mentioned_object_json_file, "r") as file_id:
        mentioned_objects = json.load(file_id)

    combine_split_lines = []

    for i in range(len(task_split_lines1)):
        task1_best_in_mentioned_object = task_split_lines1[i]['task2-mm-coref'] # 'task2-mm-coref': '< 1, 3 >',
        task1_best_not_in_mentioned_object = task_split_lines2[i]['task2-mm-coref'] # 'task2-mm-coref': '<  >',

        task1_best_in_mentioned_object_list = []
        task1_best_not_in_mentioned_object_list = []
        
        #task1_best_in_mentioned_object = '< 1, 2, 3 >'
        for object_id in object_regex.finditer(task1_best_in_mentioned_object):
                str_object_id = object_id.group(1).strip()
                int_object_id = int(str_object_id)
                task1_best_in_mentioned_object_list.append(int_object_id)
                #print(str_object_id)
        
        for object_id in object_regex.finditer(task1_best_not_in_mentioned_object):
                str_object_id = object_id.group(1).strip()
                int_object_id = int(str_object_id)
                task1_best_not_in_mentioned_object_list.append(int_object_id)
                #print(str_object_id)

        mentioned_object = mentioned_objects[i]["multimodal_context"] # List[int]

        # 根据mentioned_object重组结果

        coref_list = []

        for obj in task1_best_in_mentioned_object_list:
            if obj in mentioned_object:
                coref_list.append(obj)

        for obj in task1_best_not_in_mentioned_object_list:
            if obj not in mentioned_object and obj not in coref_list:
                coref_list.append(obj)

        # 升序排序
        coref_list = sorted(coref_list)

        coref_str = str(coref_list).replace('[', '< ').replace(']',' >') if coref_list else '<  >'

        combine_split_lines.append(
            {
                'task2-mm-coref': coref_str,
            }
        )

    return combine_split_lines



def combination_task1_result_to_one_file(
    input_line_by_line_file_for_task1_best_in_mentioned_object=None,
    input_line_by_line_file_for_task1_best_in_not_mentioned_object=None,
    mentioned_object_json_file=None):
    '''
    根据是否在mentioned_object当中选择不同的模型结果作为最终结果

    '''

    disamb_candidate_regex = re.compile(r"([A-Za-z0-9]+)")

    lines1 = read_line_by_line_file(input_line_by_line_file_for_task1_best_in_mentioned_object)
    lines2 = read_line_by_line_file(input_line_by_line_file_for_task1_best_in_not_mentioned_object)
    task_split_lines1 = convert_line_list_to_task_list(lines1)
    task_split_lines2 = convert_line_list_to_task_list(lines2)

    with open(mentioned_object_json_file, "r") as file_id:
        mentioned_objects = json.load(file_id)

    combine_split_lines = []

    for i in range(len(task_split_lines1)):
        task1_best_in_mentioned_object = task_split_lines1[i]['task1-mm-disam'] # 'task1-mm-disam': '| 1, 3 |',
        task1_best_not_in_mentioned_object = task_split_lines2[i]['task1-mm-disam'] # 'task1-mm-disam': '|  |',

        task1_best_in_mentioned_object_list = []
        task1_best_not_in_mentioned_object_list = []
        
        #task1_best_in_mentioned_object = '| 1, 2, 3 |'
        for object_id in disamb_candidate_regex.finditer(task1_best_in_mentioned_object):
                str_object_id = object_id.group(1).strip()
                int_object_id = int(str_object_id)
                task1_best_in_mentioned_object_list.append(int_object_id)
                #print(str_object_id)
        
        for object_id in disamb_candidate_regex.finditer(task1_best_not_in_mentioned_object):
                str_object_id = object_id.group(1).strip()
                int_object_id = int(str_object_id)
                task1_best_not_in_mentioned_object_list.append(int_object_id)
                #print(str_object_id)

        mentioned_object = mentioned_objects[i]["multimodal_context"] # List[int]

        # 根据mentioned_object重组结果

        disamb_list = []

        for obj in task1_best_in_mentioned_object_list:
            if obj in mentioned_object:
                disamb_list.append(obj)

        for obj in task1_best_not_in_mentioned_object_list:
            if obj not in mentioned_object and obj not in disamb_list:
                disamb_list.append(obj)

        # 升序排序
        disamb_list = sorted(disamb_list)

        disamb_str = str(disamb_list).replace('[', '| ').replace(']',' |') if disamb_list else '|  |'

        combine_split_lines.append(
            {
                'task1-mm-disam': disamb_str,
            }
        )

    return combine_split_lines



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_line_by_line_file_for_task1", default=None, type=str, help="The best results for task1")
    parser.add_argument("--input_line_by_line_file_for_task2", default=None, type=str, help="The best results for task2")
    parser.add_argument("--input_line_by_line_file_for_task3", default=None, type=str, help="The best results for task3")
    parser.add_argument("--input_line_by_line_file_for_task4", default=None, type=str, help="The best results for task4")
    parser.add_argument("--input_line_by_line_file_for_task1_best_in_mentioned_object", default=None, type=str, help="The best results for task1 in mentioned object")
    parser.add_argument("--input_line_by_line_file_for_task1_best_in_not_mentioned_object", default=None, type=str, help="The best results for task1 not in mentioned object")
    parser.add_argument("--input_line_by_line_file_for_task2_best_in_mentioned_object", default=None, type=str, help="The best results for task2 in mentioned object")
    parser.add_argument("--input_line_by_line_file_for_task2_best_in_not_mentioned_object", default=None, type=str, help="The best results for task2 not in mentioned object")
    parser.add_argument("--mentioned_object_json_file", default=None, type=str, help="The mentioned_object_json_file")
    parser.add_argument("--output_line_by_line_combination_results", default=None, type=str, help="The combination best results for task 1~4")

    parser.add_argument("--input_line_by_line_file1", default=None, type=str, help="The best results for not final turn")
    parser.add_argument("--input_line_by_line_file2", default=None, type=str, help="The best results for final turn")
    parser.add_argument("--input_path_json", default=None, type=str, help="The json input file")
    parser.add_argument("--file2_only_has_final_turn", action="store_true", help="the --input_line_by_line_file2 only has final turn")

    args = parser.parse_args()

    if args.input_line_by_line_file_for_task1 is not None and args.input_line_by_line_file_for_task2 is not None and args.input_line_by_line_file_for_task3 is not None and args.input_line_by_line_file_for_task4 is not None:
        combine_lines, combine_split_lines = combination_model_result_to_one_file(
            input_line_by_line_file_for_task1=args.input_line_by_line_file_for_task1,
            input_line_by_line_file_for_task2=args.input_line_by_line_file_for_task2,
            input_line_by_line_file_for_task3=args.input_line_by_line_file_for_task3,
            input_line_by_line_file_for_task4=args.input_line_by_line_file_for_task4
        )

    if args.input_line_by_line_file_for_task1_best_in_mentioned_object is not None and args.input_line_by_line_file_for_task1_best_in_not_mentioned_object is not None and args.mentioned_object_json_file is not None:
        combine_split_lines_task1 = combination_task1_result_to_one_file(
            input_line_by_line_file_for_task1_best_in_mentioned_object=args.input_line_by_line_file_for_task1_best_in_mentioned_object,
            input_line_by_line_file_for_task1_best_in_not_mentioned_object=args.input_line_by_line_file_for_task1_best_in_not_mentioned_object,
            mentioned_object_json_file=args.mentioned_object_json_file)

        for i in range(len(combine_split_lines)):
            combine_split_lines[i]['task1-mm-disam'] = combine_split_lines_task1[i]['task1-mm-disam']

    if args.input_line_by_line_file_for_task2_best_in_mentioned_object is not None and args.input_line_by_line_file_for_task2_best_in_not_mentioned_object is not None and args.mentioned_object_json_file is not None:
        combine_split_lines_task2 = combination_task2_result_to_one_file(
            input_line_by_line_file_for_task2_best_in_mentioned_object=args.input_line_by_line_file_for_task2_best_in_mentioned_object,
            input_line_by_line_file_for_task2_best_in_not_mentioned_object=args.input_line_by_line_file_for_task2_best_in_not_mentioned_object,
            mentioned_object_json_file=args.mentioned_object_json_file)

        for i in range(len(combine_split_lines)):
            combine_split_lines[i]['task2-mm-coref'] = combine_split_lines_task2[i]['task2-mm-coref']


    combine_lines = convert_task_list_to_line_list(combine_split_lines)

    with open(args.output_line_by_line_combination_results, "w") as f_out:
        f_out.write("\n".join(combine_lines))






