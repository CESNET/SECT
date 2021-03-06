import time, datetime, os, sys
from time import sleep
import logging
import shutil
import ast
import ipaddress
import json


class Correlations:
    """ Class for creating time correlated groups """
    def __init__(self, search_from, search_to, atacks_minimum=10,
                 time_offset=40, records_get=None, fname='./data/'):
        """ Initialize class Correlations
        :param search_from: Time from witch the search begins
        :param search_to: Time where the search ends
        :param atacks_minimum: Minimum amount of detected atacks per IP
        :param time_offset: 2 times are the same if their difference is less or
        equal to time_offset
        :param records_get: Getter for records for each IP in dict format
        """
        self.search_from = datetime.datetime.strptime(search_from, '%Y-%m-%d %H:%M:%S%z').timestamp()
        self.search_to = datetime.datetime.strptime(search_to, '%Y-%m-%d %H:%M:%S%z').timestamp()
        assert search_to > search_from

        self.time_offset = time_offset
        self.atacks_minimum = atacks_minimum
        self.time_offset = time_offset
        self.records_get = records_get or self._records_get
        self.records_for_ips = {}
        self.total_ips = 0
        self.index_sparse_matrix = {}
        self.corr_groups = []
        self.fname=fname
        self.sources = {}
        self.meta = {}

        logging.basicConfig(filename='./data/logs.log', level=logging.ERROR,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        self.logger = logging.getLogger(__name__)

    def __call__(self):
        self.records_for_ips = self.records_get(self.fname)
        self._create_index_sparse_matrixes()
        self.corr_groups = self._find_correlated_pairs()
        print("2  --  " + str(len(self.corr_groups)))
        self._find_correlated_groups(2, self.corr_groups)
        self._remove_redundancies()
        self._save_found_correlations()
        print("total correlations --> ", len(self.corr_groups))

    #todo redo to accept new format through pandas
    def _records_get(self, fname):
        """ Getter for records for each IP
        :return: dictionary in format {IP : [detected time stamps]}
        """
        records = {}
        with open(fname, "r") as f:
            self.meta = json.load(f)

            for ip, vec in self.meta['ips'].items():
                    times = [val[0] for val in vec]
                    anoted_times = [tuple([val[1], val[0]]) for val in vec]

                    # skip when activity is out of searched range
                    if (self.search_to < times[0] or times[-1] < self.search_from):
                        continue
                    records[ip] = sorted(set(times))
                    self.sources[ip] = sorted(set(anoted_times))

        new_records = {}
        for ip in records:
            times_in_interval = []
            times = records[ip]
            for time in times:
                if (self.search_from <= time <= self.search_to):
                    times_in_interval.append(time)
            if (len(times_in_interval) >= self.atacks_minimum):
                new_records[ip] = times_in_interval

        return new_records

    def _create_index_sparse_matrixes(self):
        """ Method creates index sparse matrixec from records for each IP.
        Matrixec are saved into index_sparse_matrix attribute
        """
        def create_index_sparse_matrix(times):
            """ Function creates index sparse matrix
            :param times: Time records to transform into index sparse matrix
            :return: Index sparse matrix in set
            """
            index_sparse_matrix = []
            #search_from = datetime.datetime(self.search_from, '%Y-%m-%d %H:%M:%S')
            for time in times:
                #difference = datetime.datetime.fromisoformat(time) - search_from
                #difference = int(difference.total_seconds())
                index_sparse_matrix.append(time-self.search_from)
            return set(index_sparse_matrix)

        for ip in self.records_for_ips:
            try:
                key = ipaddress.ip_address(ip)
                self.index_sparse_matrix[key] = create_index_sparse_matrix(self.records_for_ips[ip])
                self.total_ips += 1
            except (ValueError, TypeError) as err:
                self.logger.error(err)
                pass

    def _find_correlated_pairs(self):
        """ Method creates correlated pairs from index_sparse_matrix attribute
        :return: Correlated pairs
        """

        def is_correlated(matches_length, length_a, length_b):
            #todo write equivalent to check mere possibillity or correlation
            """ Function decides if two IPs are correlated
            :param matches_length: Amount of times where IPs matches
            :param length_a: Amount of detected times for the first IP
            :param length_b: Amount of detected times for the second IP
            :return: True if IPs are correlated, else False
            """
            if (matches_length == 0):
                return False
            treshold = 1.05
            if (length_a / matches_length <= treshold and length_b / matches_length <= treshold):
                return True
            return False

        copy_index_sparse_matrix = self.index_sparse_matrix.copy()
        corr_pairs = []
        for idx_sp_matrix_1 in self.index_sparse_matrix:
            del copy_index_sparse_matrix[idx_sp_matrix_1]
            for idx_sp_matrix_2 in copy_index_sparse_matrix:
                matches = []
                times_in_matrix_1 = self.index_sparse_matrix[idx_sp_matrix_1]
                times_in_matrix_2 = copy_index_sparse_matrix[idx_sp_matrix_2]

                intersection = times_in_matrix_2.intersection(times_in_matrix_1)
                if (len(intersection) > 0):
                    times_in_matrix_2 = times_in_matrix_2.difference(intersection)
                    times_in_matrix_1 = times_in_matrix_1.difference(intersection)
                    matches = list(intersection)

                # todo is it really necessary to sort this ?
                # Might be insignificant
                times_in_matrix_2 = sorted(times_in_matrix_2)
                times_in_matrix_1 = sorted(times_in_matrix_1)

                # time offset part
                for time in times_in_matrix_2:
                    for t in times_in_matrix_1:
                        if (abs(time - t) <= self.time_offset):
                            times_in_matrix_1.remove(t)
                            matches.append(t)
                            break

                if (is_correlated(len(matches),
                                  len(self.index_sparse_matrix[idx_sp_matrix_1]),
                                  len(self.index_sparse_matrix[idx_sp_matrix_2]))):
                    corr_pairs.append((set([idx_sp_matrix_1, idx_sp_matrix_2]),
                                       sorted(matches)))
        return corr_pairs

    def _count_amount(self, n):
        """ Method counts amount of suspicious occurance which is needed for
        group to be correlated
        :param n: Length of the group
        :return: Amount of suspicious occurance which is needed for the group to
        be correlated
        """
        amount = 0
        while (n > 0):
            amount += n
            n -= 1
        return amount

    # warning - super slow, big groups will make the algorithm run impossibly long
    def _find_correlated_groups(self, searching_from, corr_groups):
        """ Method finds correlated groups and saves them into corr_groups
        attribute
        :param searching_from: Length of group from which are currently searched
        groups with length Length + 1
        :param corr_groups: Correlated groups from which are search groups with
        length Length + 1
        """
        copy_corr_groups = corr_groups.copy()
        suspicious_groups = {}
        new_groups = []
        enough_amount = self._count_amount(searching_from)
        for tuple_1 in corr_groups:
            corr_IPs_1 = tuple_1[0]
            copy_corr_groups.remove(tuple_1)
            for tuple_2 in copy_corr_groups:
                corr_IPs_2 = tuple_2[0]
                intersection = corr_IPs_1.intersection(corr_IPs_2)
                if (len(intersection) != 0):
                    suspicious_group = corr_IPs_1.union(corr_IPs_2)
                    if (len(suspicious_group) == searching_from + 1):
                        key = self._create_key(suspicious_group)
                        if (key in suspicious_groups):
                            amount = suspicious_groups[key][1] + 1
                            if (amount == enough_amount):
                                new_groups.append((set(key),
                                                   suspicious_groups[key][0]))
                            else:
                                occurrences = suspicious_groups[key][0]
                                suspicious_groups[key] = (occurrences, amount)
                        else:
                            suspicious_groups[key] = (tuple_1[1], 1)

        print(searching_from + 1, " -- ", len(new_groups))

        if (len(new_groups) == 0):
            return

        self.corr_groups += new_groups

        searching_from += 1
        if (searching_from < self.total_ips):
            self._find_correlated_groups(searching_from, new_groups)
        else:
            return

    def _create_key(self, groups):
        """ Methods creates hashable key from set of IPs
        :param groups: set of IPs
        :return: Hashable key
        """
        ipv4 = []
        ipv6 = []
        for ip in groups:
            if (ip.version == 4):
                ipv4.append(ip)
            else:
                ipv6.append(ip)

        key = sorted(ipv4 + ipv6)
        return tuple(key)

    def _remove_redundancies(self):
        """ Method removes redundancies from corr_groups attribute
        """
        copy_corr_groups = self.corr_groups.copy()
        new_corr_groups = []
        for tuple_1 in self.corr_groups:
            copy_corr_groups.remove(tuple_1)
            flag = True
            length = len(tuple_1[0])
            for tuple_2 in copy_corr_groups:
                if (len(tuple_1[0].intersection(tuple_2[0])) == length):
                    flag = False
                    break
            if (flag):
                new_corr_groups.append(tuple_1)
        self.corr_groups = new_corr_groups

    def _save_found_correlations(self):
        """ Method links detectors which have detected the correlations and
        writes all these informations into files in correlations folder
        """
        sources = self.sources

        if (os.path.isdir(self.fname+"_correlations")):
            shutil.rmtree(self.fname+"_correlations", ignore_errors=True)
            sleep(.0000000000000001)  # OS Windows sometimes keeps lock on removed dir
        try:
            os.mkdir(self.fname+"_correlations")
        except PermissionError as err:
            sys.stderr.write("Can not create dir correlations - Permission denied.\n")
            self.logger.error(err)
            logging.shutdown()
            sys.exit(1)

        #search_from = datetime.datetime.strptime(self.search_from, '%Y-%m-%d %H:%M:%S')
        for correlation in self.corr_groups:
            path = self.fname+"_correlations/"
            concat_1 = ""
            concat_2 = "IPs (" + str(len(correlation[0])) + "):\n"
            for ip in correlation[0]:
                concat_1 += str(ip) + "&"
                concat_2 += str(ip) + "\n"
            concat_1 = concat_1[:-1]
            path += concat_1 + ".txt"
            file = open(path, 'w')
            write = concat_2 + "\nTimes (" + str(len(correlation[1])) + "):\n"
            times = []
            for time in correlation[1]:
                #tmp = datetime.timedelta(seconds=time)
                tmp = self.search_from + time
                times.append(tmp)
                write += str(datetime.datetime.fromtimestamp(tmp)) + ", "
            file.write(write)

            file.write("\n\nDetectors:\n")
            detectors = {}
            for ip in concat_1.split("&"):
                source = sources[ip][1:].copy()
                flag = True
                while (flag):
                    flag = False
                    for pair in source:
                        if (flag):
                            break
                        for time in times:
                            if (time == pair[1]):
                                if (pair[0] in detectors):
                                    detectors[pair[0]].append(time)
                                else:
                                    detectors[pair[0]] = [time]
                                source.remove(pair)
                                flag = True
                                break

            for detector in detectors:
                file.write(str(self.meta['origins'][str(detector)]) + " --> " +
                           str([datetime.datetime.fromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S') for val in sorted(detectors[detector])]) + "\n")
                #file.write(str(detector) + " --> " + str(sorted(detectors[detector])) + "\n")

            file.close()


if __name__ == '__main__':
    start_time = time.time()
    #korelace = Correlations("2019-03-11 00:30:00", "2019-03-11 01:30:00", atacks_minimum=2, fname='./data/2019-03-11')
    date = '2019-08-01'
    korelace = Correlations(date+" 01:30:00Z", date+" 01:59:00Z", atacks_minimum=10, time_offset=15,  fname='./data/'+date+'_prep_.json')

    korelace()

    print("--- %s seconds ---" % (time.time() - start_time))