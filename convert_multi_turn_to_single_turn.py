filename = "llava_v1_5_mix665k"

import os
from copy import deepcopy
def main():
    import json
    file = f"./playground/data/{filename}.json"
    samples = json.load(open(file, 'r'))
    print(f"Total samples: {len(samples)}")
    
    single_turn_count = 0
    multi_turn_count = 0
    converted_multi_turn_samples = []
    for sample in samples:
        if len(sample["conversations"]) == 2:
            single_turn_count += 1
            converted_multi_turn_samples.append(sample)
        else:
            multi_turn_count += 1
            
            first_turn_conversation = sample["conversations"][:2]
            first_turn_sample = deepcopy(sample)
            first_turn_sample["conversations"] = first_turn_conversation
            converted_multi_turn_samples.append(first_turn_sample)
            
            
            for i in range(2, len(sample["conversations"]), 2):
                new_turn_sample = deepcopy(sample)
                new_turn_conversation = deepcopy(new_turn_sample["conversations"][i:i+2])
                new_turn_conversation[0]['value'] = "<image>\n" + new_turn_conversation[0]['value']
                new_turn_sample["conversations"] = new_turn_conversation
                converted_multi_turn_samples.append(new_turn_sample)
            
    print(f"Total single-turn samples: {single_turn_count}")
    print(f"Total multi-turn samples: {multi_turn_count}")
    print(f"Total converted multi-turn samples: {len(converted_multi_turn_samples)}")
    new_file = f"./playground/data/{filename}_flattened_multi_turn.json"
    json.dump(converted_multi_turn_samples, open(new_file, 'w'), indent=4)
    print(f"Saved to {new_file}")
    
if __name__ == "__main__":
    main()