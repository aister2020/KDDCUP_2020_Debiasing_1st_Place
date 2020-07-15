from constants import *
import utils

cur_stage = CUR_STAGE

def item2time():
    for mode in ["valid", "test"]:
        if mode=='valid':
            all_train_data = utils.load_pickle(all_train_data_path.format(cur_stage))
        else:
            all_train_data = utils.load_pickle(online_all_train_data_path.format(cur_stage))

        item_with_time = all_train_data[["item_id", "time"]].sort_values(["item_id", "time"])
        item2time = item_with_time.groupby("item_id")["time"].agg(list).to_dict()
        utils.dump_pickle(item2time, item2time_path.format(mode, cur_stage))

def item_pair2time_seq():
    for mode in ["valid", "test"]:
        if mode=='valid':
            all_train_data = utils.load_pickle(all_train_data_path.format(cur_stage))
        else:
            all_train_data = utils.load_pickle(online_all_train_data_path.format(cur_stage))

        all_item_seqs = all_train_data.groupby("user_id")["item_id"].agg(list).tolist()
        all_time_seqs = all_train_data.groupby("user_id")["time"].agg(list).tolist()

        # 固定时间间隔内共现的次数统计，多种粒度
        # 两个item共现时的time_diff都是哪些，维护一个sequence
        item_pair2time_diff = {}
        item_pair2time_seq = {}
        cnt = 0
        for item_seq, time_seq in zip(all_item_seqs, all_time_seqs):
            length = len(item_seq)
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
            for i in range(length):
                for j in range(i+1, length):
                    itemA, itemB = item_seq[i], item_seq[j]
                    timeA, timeB = time_seq[i], time_seq[j]
                    time_diff = abs(timeA - timeB)
                    times = tuple(sorted([timeA, timeB]))
                    pair = tuple(sorted([itemA, itemB]))

                    item_pair2time_seq.setdefault(pair, [])
                    item_pair2time_seq[pair].append(times)

                    item_pair2time_diff.setdefault(pair, [])
                    item_pair2time_diff[pair].append(time_diff)

        utils.dump_pickle(item_pair2time_seq, item_pair2time_seq_path.format(mode, cur_stage))
        utils.dump_pickle(item_pair2time_diff, item_pair2time_diff_path.format(mode, cur_stage))

def item_pair2time_diff():
    for mode in ["valid", "test"]:
        if mode=='valid':
            all_train_data = utils.load_pickle(all_train_data_path.format(cur_stage))
        else:
            all_train_data = utils.load_pickle(online_all_train_data_path.format(cur_stage))

        all_item_seqs = all_train_data.groupby("user_id")["item_id"].agg(list).tolist()
        all_time_seqs = all_train_data.groupby("user_id")["time"].agg(list).tolist()

        # 固定时间间隔内共现的次数统计，多种粒度
        # 两个item共现时的time_diff都是哪些，维护一个sequence
        deltas = [0.01, 0.03, 0.05, 0.07, 0.1]
        item2times = {}
        # item_pair2times = {}
        for delta in deltas:
            item2times[delta] = {}
        cnt = 0
        for item_seq, time_seq in zip(all_item_seqs, all_time_seqs):
            length = len(item_seq)
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
            for i in range(length):
                for j in range(i+1, length):
                    itemA, itemB = item_seq[i], item_seq[j]
                    timeA, timeB = time_seq[i], time_seq[j]
                    time_diff = abs(timeA - timeB)
                    pair = tuple(sorted([itemA, itemB]))
                    # item_pair2times.setdefault(pair, [])
                    # item_pair2times[pair].append(time_diff)
                    for delta in deltas:
                        if time_diff < delta:
                            item2times[delta].setdefault(pair, 0)
                            item2times[delta][pair] += 1

        # for pair in item_pair2times:
        #     item_pair2times[pair] = sorted(item_pair2times[pair])
        for delta in deltas:
            utils.dump_pickle(item2times, item2times_path.format(mode, cur_stage, delta))
        # utils.dump_pickle(item_pair2times, item_pair2times_path.format(mode, cur_stage))

if __name__ == "__main__":
    item2time()
    item_pair2time_diff()
    item_pair2time_seq()