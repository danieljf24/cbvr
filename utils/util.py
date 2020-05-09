import os
import csv

def read_video_set(filepath):
    reader = csv.reader(open(filepath, 'r'))
    data = []
    for x in reader:
        data.append(x[0])
    return data	


def write_csv(filepath, data):
    """
    wirete csv file.
    :param file_path: the path of the csv file
    :param data: writing data
    """
    with open(filepath, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        for line in data:
            csv_writer.writerow(line)


def read_csv_video2rank(filepath):
    video2rank = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for data in reader:
            video = data[0]
            assert video not in video2rank
            video2rank[video] = data[1:]
    return video2rank


def write_csv_video2rank(filepath, video2rank, topk=500):
    result_data = []
    for video, ranks in video2rank.items():
        result_data.append([video] + ranks[:topk])
    write_csv(filepath, result_data)


def write_csv_video2rank_fusion(videolist, filepath, video2rank, topk=500):
    result_data = []
    assert len(videolist) == len(video2rank)
    for video in videolist:
        ranks = video2rank[video]
        result_data.append([video] + ranks[:topk])
    write_csv(filepath, result_data)


def read_dict(filepath):
    f = open(filepath,'r')  
    a = f.read()  
    dict_data = eval(a)  
    f.close()
    return dict_data


def write_dict(filepath, dict_data):
    f = open(filepath,'w')  
    f.write(str(dict_data))  
    f.close()


def get_count(path):
    train=os.path.join(path,'split/train.csv')
    val= os.path.join(path,'split/val.csv')
    id_list=[]
    train_reader=csv.reader(open(train))
    val_reader=csv.reader(open(val))

    for vid in train_reader:
        id_list.append(vid[0])
    for vid in val_reader:
        id_list.append(vid[0])
    id_set = set(id_list)

    rele_train=os.path.join(path,'relevance_train.csv')
    rele_val = os.path.join(path,'relevance_val.csv')
    rele_train_reader=csv.reader(open(rele_train))
    rele_val_reader=csv.reader(open(rele_val))
    
    output_file = os.path.join(path, 'rel_index.csv')
    with open(output_file, 'w') as csvfile:
        writew = csv.writer(csvfile)
        for rele in rele_train_reader:
            index_list=[]
            for rele_id in (rele[1:]):
                if rele_id in id_set:
                    index_list.append(id_list.index(rele_id))
            writew.writerow(index_list)

        for rele in rele_val_reader:
            index_list=[]
            for rele_id in rele[1:]:
                if rele_id in id_set:
                    index_list.append(id_list.index(rele_id))
            writew.writerow(index_list)

    print('write out: %s' % output_file)