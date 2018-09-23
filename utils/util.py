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