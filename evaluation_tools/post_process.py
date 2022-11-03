import sys
import re
import pdb


def correct_available_sizes(text):
    SIZES =['<A>', '<B>', '<C>', '<D>', '<E>', '<F>']
    try:
        if 'availableSizes =' in text:
            available_sizes_str_list = [(m.start(0), m.end(0)) for m in re.finditer(r"availableSizes =", text)]
            if not available_sizes_str_list:  # empty available_sizes_str_list: in case of (availableSizes)
                return text
            availableSizes_idx = available_sizes_str_list[0][1]
            start_bracket_idx = -1
            end_bracket_idx = -1
            for i in range(100):
                if text[availableSizes_idx+i] == '[':
                    start_bracket_idx = availableSizes_idx+i
                if text[availableSizes_idx+i] == ']':
                    end_bracket_idx = availableSizes_idx+i
                if start_bracket_idx != -1 and end_bracket_idx != -1:
                    break
            assert start_bracket_idx != -1 and end_bracket_idx != -1, f"ERROR AT def correct_available_sizes!!\n{text}"
            list_str = text[start_bracket_idx:end_bracket_idx].replace("'", "")
            new_list = []
            for size in SIZES:
                if size in list_str:
                    new_list.append(size)
            new = ", ".join(new_list)
            return text[:start_bracket_idx] + '['+new + text[end_bracket_idx:]
        else:
            return text
    except:
        return text
        print('text:', text)


def replace_special_chars(text):
    def rep(match_re_obj):
        return match_re_obj.group(0).replace('<','').replace('>','')
    available_sizes_st_list = [('<A>', "'XS'"), ('<B>', "'S'"), ('<C>', "'M'"), ('<D>', "'L'"), ('<E>', "'XL'"), ('<F>', "'XXL'")]
    for size_tuple in available_sizes_st_list:
        text = text.replace(size_tuple[0], size_tuple[1])
    text = re.sub("<[0-9]+>", rep, text)
    return text


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        
        items = line.split("\t")
        gens, coref_objs, disamb_objs = items[:]

        gens = correct_available_sizes(gens)
        gens = replace_special_chars(gens)
        print("\t".join([gens, coref_objs, disamb_objs]))

        # line = correct_available_sizes(line)
        # line = replace_special_chars(line)
        # print(line)

    # with open("~/projects/19_dstc11/simmc2.1_solutions/work/simmc2.1-scut-bds-lab/train_v1/1_train_model_simmc21_bart_large_20220421/test/epoch0/devtest.out", "r") as fr:
    #     for line in fr.readlines():
    #         line = line.strip()
    #         try:
    #             xline = correct_available_sizes(line)
    #             xline = replace_special_chars(xline)
    #         except:
    #             pdb.set_trace()
    #             xline = correct_available_sizes(line)
    #         # print(line)