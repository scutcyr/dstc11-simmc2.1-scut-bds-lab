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

    return combine_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_line_by_line_file_for_task1", default=None, type=str, required=True, help="The best results for task1")
    parser.add_argument("--input_line_by_line_file_for_task2", default=None, type=str, required=True, help="The best results for task2")
    parser.add_argument("--input_line_by_line_file_for_task3", default=None, type=str, required=True, help="The best results for task3")
    parser.add_argument("--input_line_by_line_file_for_task4", default=None, type=str, required=True, help="The best results for task4")
    parser.add_argument("--output_line_by_line_combination_results", default=None, type=str, required=True, help="The combination best results for task 1~4")
    args = parser.parse_args()

    combine_lines = combination_model_result_to_one_file(
        input_line_by_line_file_for_task1=args.input_line_by_line_file_for_task1,
        input_line_by_line_file_for_task2=args.input_line_by_line_file_for_task2,
        input_line_by_line_file_for_task3=args.input_line_by_line_file_for_task3,
        input_line_by_line_file_for_task4=args.input_line_by_line_file_for_task4
    )

    with open(args.output_line_by_line_combination_results, "w") as f_out:
        f_out.write("\n".join(combine_lines))


