import json, codecs
import numpy as np

import pandas as pd
# pd.Series(your_array).to_json(orient='values')

def read_json(filename):
  with open(filename) as json_file:
      return json.load(json_file)

def write_json(filename, data):
  with open(filename, 'w') as outfile:
    json.dump(data, outfile, indent=2)

if __name__=="__main__":
    data = {}

    data['beacons'] = []

    data['beacons'].append({
        'Type': 'SCS',
        'ID': '0',
        'Pathtree': [1,2,3,4,5],
        'pos_traj': np.array([np.array([1,2]).reshape(2,1),np.array([3,4]).reshape(2,1)]).tolist(),
        'force_traj': np.array([np.array([7,8]).reshape(2,1),np.array([9,10]).reshape(2,1)]).tolist(),
        'heading_traj': np.array([np.array([5,5]).reshape(2,1),np.array([6,6]).reshape(2,1)]).tolist(),
        'xi_traj': np.array([1,2,3,4,5,6]).tolist()
        #Vectors and stuff

    })
        
    test = pd.Series(np.array([np.array([1,2]).reshape(2,1), np.array([4,5]).reshape(2,1)]).tolist()).to_json(orient='split')
    # print(f"DEtte: {test}")
    data['beacons'].append({
        'Type': 'MIN',
        'ID': '1',
        'Pathtree': [1,2,3,4,5],
        'pos_traj': np.array([np.array([1,2]).reshape(2,1)]).tolist(),#,np.array([3,4]).reshape(2,1)])).to_json(orient='values')#,
        'force_traj': np.array([np.array([7,8]).reshape(2,1),np.array([9,10]).reshape(2,1)]).tolist(),
        'heading_traj': np.array([np.array([5,5]).reshape(2,1),np.array([6,6]).reshape(2,1)]).tolist(),
        'xi_traj': np.array([1,2,3,4,5,6]).tolist()
        #Vectors and stuff
    })

    data['beacons'].append({
        'Type': 'MIN',
        'ID': '2',
        'Pathtree': [1,2,3,4,5],
        'pos_traj': np.array([np.array([1,2]).reshape(2,1),np.array([3,4]).reshape(2,1)]).tolist(),
        'force_traj': np.array([np.array([7,8]).reshape(2,1),np.array([9,10]).reshape(2,1)]).tolist(),
        'heading_traj': np.array([np.array([5,5]).reshape(2,1),np.array([6,6]).reshape(2,1)]).tolist(),
        'xi_traj': np.array([1,2,3,4,5,6]).tolist()
        #Vectors and stuff
    })

# file_path = "/path.json" ## your path variable
# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
with open('data_1.txt', 'w') as outfile:
    json.dump(data, outfile)

# # "Unjsonify" from top answer at https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
obj_text = codecs.open('data_1.txt', 'r', encoding='utf-8').read()
b_new = json.loads(obj_text),
# print(f"b_new: {b_new}")
print(f"{b_new[0]['beacons']['Type']}")
# a_new = np.array(b_new)
# print(f"a_new: {a_new}")