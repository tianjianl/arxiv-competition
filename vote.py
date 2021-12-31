import csv
from collections import Counter

def vote_merge(filelst):
    result = {}
    fw = open('merge.csv', encoding='utf-8', mode='w', newline='')
    csv_writer = csv.writer(fw)
    csv_writer.writerow(['nid', 'label'])
    for filepath in filelst:
        cr = open(filepath, encoding='utf-8', mode='r')
        csv_reader = csv.reader(cr)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            idx, cls = row
            if idx not in result:
                result[idx] = []
            result[idx].append(cls)

    for nid, clss in result.items():
        counter = Counter(clss)
        true_cls = counter.most_common(1)
        csv_writer.writerow([nid, true_cls[0][0]])

if __name__ == '__main__':
    vote_merge([
        'submission_cs (3).csv',
        'submission_cs (4).csv',
        'submission_cs (5).csv',
        'submission_cs (6).csv',
        'submission_cs (7).csv',
        'submission_cs (8).csv',
        'submission_cs (9).csv',
        'submission_cs (10).csv',
        'submission_cs (11).csv'
                ])