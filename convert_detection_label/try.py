import json


def save_dict_to_json_file(json_file_path, dic):
    with open(json_file_path, "w") as fp:
        json.dump(dic, fp)


if __name__ == "__main__":
    class_list = "Logo.names"

    with open(class_list, 'r') as f:
        labels = f.readlines()

    # 1. Create the dictionary of logo namess
    dic = {x.strip().split("\n")[0]: i for i, x in enumerate(labels)}

    # 2. Create list of logo names
    for i, x in enumerate(labels):
        labels[i] = x.strip().split("\n")[0]  # get the class name itself out.

    print(dic)
    print()
    print(labels)

    logo_dic_file_name = "Logo.json"
    save_dict_to_json_file(logo_dic_file_name, dic)
