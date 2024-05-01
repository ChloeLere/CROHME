import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm

path = './'
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

SEED = 999
seed_everything(SEED)

def get_traces_data(inkml_file_abs_path):
    traces_data = []
    try:
        tree = ET.parse(inkml_file_abs_path)
    except ET.ParseError as e:
        print(f"Error parsing file {inkml_file_abs_path}: {e}")
        return None
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    #'Stores traces_all with their corresponding id'
    traces_all = [{'id': trace_tag.get('id'),
    					'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
    								else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord.split(' ')] \
    							for coord in (trace_tag.text).replace('\n', '').split(',')]} \
    							for trace_tag in root.findall(doc_namespace + 'trace')]
    #'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
    #'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            label = traceGroup.find(doc_namespace + 'annotation').text
            #'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                #'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))
                #'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)
            traces_data.append({'label': label, 'trace_group': traces_curr})
    else:
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    return traces_data

def inkml2img(input_path, output_path):
    traces = get_traces_data(input_path)
    if traces == None:
        return
    path = input_path.split('/')
    path = path[len(path)-1].split('.')
    path = path[0]+'_'
    file_name = 0
    for elem in traces:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.invert_yaxis()

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Hide the spines (borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ls = elem['trace_group']
        output_path = output_path  
        
        for subls in ls:
            data = np.array(subls)
            x,y=zip(*data)
            plt.plot(x,y,linewidth=2,c='black')
        capital_list = ['A','B','C','F','X','Y']
        if elem['label'] in capital_list:
            label = 'capital_'+elem['label']
        else:
            label = elem['label']
        ind_output_path = output_path + label
        try:
            os.mkdir(ind_output_path)
        except OSError:
            pass
        else:
            pass
        if(os.path.isfile(ind_output_path+'/'+path+str(file_name)+'.png')):
            file_name += 1
            plt.savefig(ind_output_path+'/'+path+str(file_name)+'.png', bbox_inches='tight', dpi=100)
        else:
            plt.savefig(ind_output_path+'/'+path+str(file_name)+'.png', bbox_inches='tight', dpi=100)
        plt.gcf().clear()
        plt.close(fig)

for filename in ["/trainData_2012_part1", "/trainData_2012_part2","/MatricesTrain2014"]:#2011 2012 2012_2 2013, 2014["/TrainINKML_2013", "/CROHME_training_2011", "/trainData_2012_part1", "/trainData_2012_part2", "/MatricesTrain2014"]:
    files = os.listdir(path+filename)
    print("=====================================", filename, "=====================================")
    for file in tqdm(files):
        if file.endswith('.inkml'):
            inkml2img(path+filename+'/'+file,'./finaltrain/')
            os.remove(path+filename+'/'+file)
    
files = os.listdir(path+"/MatricesTest2014")
for file in tqdm(files):
    if file.endswith('.inkml'):
        inkml2img(path+'/MatricesTest2014/'+file,'./finaltest/')