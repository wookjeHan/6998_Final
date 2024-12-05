from .commonsense_constraint import evaluation as commonsense_eval
from .hard_constraint import evaluation as hard_eval
import json
from tqdm import tqdm
from datasets import load_dataset


def load_line_json_data(filename):
    print("FILENAME : ", filename)
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def count_true_false(data):
    """Count the number of true and false values in a list."""
    true_count = data.count(True)
    false_count = data.count(False)
    return true_count, false_count

def statistics(commonsense_statistic):
    """Generate statistics for each level and day in the given data with a different structure."""
    result = {level: {day: {} for day in commonsense_statistic[level]} for level in commonsense_statistic}
    
    for level, days in commonsense_statistic.items():
        for day, dicts in days.items():
            for dct in dicts:
                if dct:
                    for key, data in dct.items():
                        true_count, false_count = count_true_false(data)
                        if key not in result[level][day]:
                            result[level][day][key] = {"true": 0, "false": 0}
                        result[level][day][key]["true"] += true_count
                        result[level][day][key]["false"] += false_count
                
    return result

def paper_term_mapping(commonsense_constraint_record, hard_constraint_record):
    mapping_dict = {'is_valid_information_in_current_city':'Within Current City','is_valid_information_in_sandbox':'Within Sandbox','is_reasonalbe_visiting_city':'Reasonable City Route','is_valid_restaurants':'Diverse Restaurants','is_valid_transportation':'Non-conf. Transportation','is_valid_attractions':'Diverse Attractions','is_valid_accommodation':'Minimum Nights Stay','is_not_absent':'Complete Information','valid_cost':'Budget','valid_room_rule':'Room Rule','valid_cuisine':'Cuisine','valid_room_type':'Room Type','valid_transportation':'Transportation'}
    remap_commonsense_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    remap_hard_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_commonsense_constraint_record[level][day] = {mapping_dict[key] : val for key,val in commonsense_constraint_record[level][day].items()}
            remap_hard_constraint_record[level][day] = {mapping_dict[key] : val for key,val in hard_constraint_record[level][day].items()}
    return remap_commonsense_constraint_record, remap_hard_constraint_record


def eval_score(set_type: str, file_path: str, is_debug: bool):

    if set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train',download_mode="force_redownload")['train']
    elif set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation',download_mode="force_redownload")['validation']

    
    query_data_list = [x for x in query_data_list][:20] if is_debug else [x for x in query_data_list]
    hardConstraint_statistic= {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    commonsenseConstraint_statistic = {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    tested_plans = load_line_json_data(file_path)
    delivery_cnt = 0
    plan_constraint_store = []
    for idx in tqdm(range(0,len(query_data_list))):
        query_data = query_data_list[idx]
        tested_plan = tested_plans[idx]
        if type(query_data) == str:
            query_data = eval(query_data)
        if type(tested_plan) == str:
            tested_plan = eval(tested_plan)
        if type(query_data['local_constraint']) == str:
            query_data['local_constraint'] = eval(query_data['local_constraint'])

        if tested_plan['plan']:
            delivery_cnt += 1
            commonsense_info_box = commonsense_eval(query_data,tested_plan['plan'])
            
        else:
            commonsense_info_box = None

        if commonsense_info_box and commonsense_info_box['is_not_absent'][0] and commonsense_info_box['is_valid_information_in_sandbox'][0]:
            hard_info_box = hard_eval(query_data,tested_plan['plan'])
        else:
            hard_info_box = None

        plan_constraint_store.append({'commonsense_constraint':commonsense_info_box,'hard_constraint':hard_info_box})

        commonsenseConstraint_statistic[query_data['level']][query_data['days']].append(commonsense_info_box)
        hardConstraint_statistic[query_data['level']][query_data['days']].append(hard_info_box)
        
    constraint_record = {key: {day: {'house rule':0, 'cuisine':0, 'room type':0, 'transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    constraint_mapping = {'house rule':'valid_room_rule','cuisine':'valid_cuisine','room type':'valid_room_type','transportation':'valid_transportation'}
    mapping_constraint_record = {key: {day: {'valid_room_rule':0, 'valid_cuisine':0, 'valid_room_type':0, 'valid_transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    count_record = {key:{day:0 for day in [3,5,7]} for key in ['easy','medium','hard']}

    for unit in query_data_list:
        count_record[unit['level']][unit['days']] += 1
        for key in constraint_record['medium'][3]:
            if unit['local_constraint'][key] != None:
                constraint_record[unit['level']][unit['days']][key] += 1
                mapping_constraint_record[unit['level']][unit['days']][constraint_mapping[key]] += 1

    commonsenseConstraint_statistic_processed = statistics(commonsenseConstraint_statistic)
    hardConstraint_statistic_processed = statistics(hardConstraint_statistic)

    
    data_record = {key:{day:[] for day in [3,5,7]} for key in ['easy','medium','hard']}

    constraint_dis_record = {"commonsense":{"pass":0,"total":0},"hard":{"pass":0,"total":0}}
    constraint_count = {key:{day:{} for day in [3,5,7]} for key in ['easy','medium','hard']}

    for constraint in ['commonsense','hard']:
        if constraint == 'commonsense':
            constraint_statistic = commonsenseConstraint_statistic_processed
        elif constraint == 'hard':
            constraint_statistic = hardConstraint_statistic_processed

        key_dict = {'commonsense':['is_valid_information_in_current_city','is_valid_information_in_sandbox','is_reasonalbe_visiting_city','is_valid_restaurants','is_valid_transportation','is_valid_attractions','is_valid_accommodation','is_not_absent'],'hard':['valid_cost','valid_room_rule','valid_cuisine','valid_room_type','valid_transportation']}
        
        for key in constraint_statistic:
            for key2 in constraint_statistic[key]:
                if key2 == -1:
                    print(constraint_statistic[key])
                    exit(0)
                for key3 in key_dict[constraint]:
                    data_record[key][key2].append('0/0')
                    if key3 in constraint_statistic[key][key2]:
                        constraint_dis_record[constraint]['pass'] += constraint_statistic[key][key2][key3]['true']
                        if constraint == 'hard':
                            if key == 'hard' and key3 in ['valid_room_rule','valid_cuisine','valid_room_type','valid_transportation']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            elif key == 'medium' and key3 in ['valid_room_rule','valid_cuisine','valid_room_type']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            else:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                                if key3 in ['valid_cost','valid_visitng_city_number','valid_days']:
                                    constraint_dis_record[constraint]['total'] += count_record[key][key2]
                                    constraint_count[key][key2][key3] = count_record[key][key2]
                                    hardConstraint_statistic_processed[key][key2][key3]['total'] = count_record[key][key2]
                        else:
                            data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                            constraint_dis_record[constraint]['total'] += count_record[key][key2]
                            constraint_count[key][key2][key3] = count_record[key][key2]
                            commonsenseConstraint_statistic_processed[key][key2][key3]['total'] =  count_record[key][key2]
    
    return commonsenseConstraint_statistic_processed, hardConstraint_statistic_processed


def eval_fn(set_type, file_path, eval_path, is_debug=False):
    common, hard = eval_score(set_type, file_path, is_debug)
    return common, hard
    