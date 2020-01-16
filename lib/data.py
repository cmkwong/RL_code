import os
import csv
import glob
import numpy as np
import collections


Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

def float_available(f):
    if f != "null":
        return float(f)
    else:
        return float(0)

class csv_reader:
    def __init__(self):
        self.total_count_filter = 0
        self.total_count_out = 0
        self.total_count_fixed = 0

    def read_csv(self, file_name, sep=',', filter_data=True, fix_open_price=False):
        print("Reading", file_name)
        with open(file_name, 'rt', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter=sep)
            h = next(reader)
            if 'Open' not in h and sep == ',':
                return self.read_csv(file_name, ';')
            indices = [h.index(s) for s in ('Open', 'High', 'Low', 'Close', 'Volume')]
            o, h, l, c, v = [], [], [], [], []
            count_out = 0
            count_filter = 0
            count_fixed = 0
            prev_vals = None
            for row in reader:
                vals = list(map(float_available, [row[idx] for idx in indices]))
                if filter_data and ((vals[-1] < (1e-8))): # filter out the day when no volume
                    count_filter += 1
                    continue

                po, ph, pl, pc, pv = vals

                # fix open price for current bar to match close price for the previous bar
                if fix_open_price and prev_vals is not None:
                    ppo, pph, ppl, ppc, ppv = prev_vals
                    if abs(po - ppc) > 1e-8:
                        count_fixed += 1
                        po = ppc
                        pl = min(pl, po)
                        ph = max(ph, po)
                count_out += 1
                o.append(po)
                c.append(pc)
                h.append(ph)
                l.append(pl)
                v.append(pv)
                prev_vals = vals
        print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
            count_filter + count_out, count_filter, count_fixed))
        # stored data
        self.total_count_filter += count_filter
        self.total_count_out += count_out
        self.total_count_fixed += count_fixed
        return Prices(open=np.array(o, dtype=np.float32),
                      high=np.array(h, dtype=np.float32),
                      low=np.array(l, dtype=np.float32),
                      close=np.array(c, dtype=np.float32),
                      volume=np.array(v, dtype=np.float32))


def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    # rv = (prices.volume) / np.mean(prices.volume)
    return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

class Simplespliter:
    def __init__(self):
        self.trainSet_size = 0
        self.valSet_size = 0

    def split_data(self, prices, percentage=0.8):
        offset = np.int(prices.high.shape[0] * percentage)
        train_data = Prices(open=prices.open[:offset], high=prices.high[:offset], low=prices.low[:offset],
                            close=prices.close[:offset], volume=prices.volume[:offset])
        eval_data = Prices(open=prices.open[offset:], high=prices.high[offset:], low=prices.low[offset:],
                            close=prices.close[offset:], volume=prices.volume[offset:])

        print("Split data done, training data: %d rows, eval data: %d" %
              (train_data.high.shape[0], eval_data.high.shape[0]))
        self.trainSet_size += train_data.high.shape[0]
        self.valSet_size += eval_data.high.shape[0]
        return train_data, eval_data

def load_relative(csv_file):
    reader = csv_reader()
    return prices_to_relative(reader.read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result

def load_fileList(path):
    file_list = os.listdir(path)
    return file_list, path

def read_bundle_csv(path, sep=',', filter_data=True, fix_open_price=False, percentage=0.8):
    reader = csv_reader()
    spliter = Simplespliter()
    train_set_dicts = {}
    eval_set_dicts = {}
    file_list = os.listdir(path)
    for file_name in file_list:
        required_path = path + "/" + file_name
        prices = reader.read_csv(required_path, sep=sep, filter_data=filter_data, fix_open_price=fix_open_price)
        train_set, val_set = spliter.split_data(prices, percentage=percentage)
        train_set_dicts[file_name] = train_set
        eval_set_dicts[file_name] = val_set
    print("Totally, read done, got %d rows, %d filtered, %d open prices adjusted" % (
        reader.total_count_filter + reader.total_count_out, reader.total_count_filter, reader.total_count_fixed))
    print("The whole data set size for training: %d and for evaluation: %d" %(
        spliter.trainSet_size, spliter.valSet_size))
    return train_set_dicts, eval_set_dicts
