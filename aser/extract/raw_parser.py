import os
import math
import errno
import json
from typing import List
from multiprocessing import Pool
from shutil import copyfile
from tqdm import tqdm
from .utils import get_corenlp_client, parse_sentense_with_stanford
from .entity_linker import LinkSharedSource, Mention, Entity, str_contain, acronym, DisjointSet, base_url, link
from ..utils.config import get_raw_process_parser


# ------------------- class section -------------------
class FileName:
    def __init__(self, root, fn):
        self.root = root
        self.fn = fn
        self.full = os.path.join(root, fn)

    def __str__(self):
        return '{} {}'.format(self.root, self.fn)

    def __repr__(self):
        return self.__str__()


class ParsingTask:
    def __init__(self, file_list: List[FileName], parsed_root, corenlp_path, port, annotators, link_flg=True):
        self.file_list = file_list
        self.corenlp_path = corenlp_path
        self.port = port
        self.annotators = annotators
        self.parsed_root = parsed_root
        self.link_flg = link_flg


# ------------------- utils section -------------------
def read_raw(file_name):
    with open(file_name) as f:
        raw = [l.strip() for l in f.readlines()]
    return raw


def silent_remove(file_name: str):
    try:
        os.remove(file_name)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def read_dir(root):
    def read_dir_func(folder):
        flist = []
        for f in os.listdir(folder):
            tmp = folder[len(root) + 1:]
            fn = FileName(root, os.path.join(tmp, f))
            if os.path.isfile(fn.full):
                flist.append(fn)
            else:
                flist.extend(read_dir_func(fn.full))
        return flist

    files = read_dir_func(root)
    return files


def dump_paths(name: str, fn_list: List[FileName]):
    with open(name, 'w') as fw:
        for fn in fn_list:
            fw.write(f"{fn.full}\n")


def load_paths(name: str, root: str = None):
    tmp, file_name_list = [], []
    for line in open(name, 'r'):
        line = line.strip()
        tmp.append(line)
    if root is not None:
        for path in tmp:
            fn = FileName(root, path[len(root) + 1:])
            file_name_list.append(fn)
        return file_name_list
    return tmp


def change_file_extension(fn, new_extension='jsonl'):
    fs = fn.split('.')[:-1]
    fs.append(new_extension)
    return '.'.join(fs)


def check_func(task):
    parsed_num, unparsed_num, empty_num = 0, 0, 0
    file_list = task.file_list
    parsed_root = task.parsed_root
    check_unparsed_rawlist = []

    def check_file_empty(fn: str):
        try:
            for _ in open(fn):
                return False
        except BaseException as e:
            print(f'{fn} error: {e}')
            return False
        return True

    def check_file_integrity(fn: str):
        lens = None
        line_num, except_line_num = 0, None
        for line in open(fn):
            line = line.strip()
            if lens is None:
                lens = json.loads(line)['sentence_lens']
                except_line_num = lens[-1]
            line_num += 1
        if except_line_num == line_num:
            return True
        return False

    parsed_fn_list = [os.path.join(parsed_root, change_file_extension(f.fn)) for f in file_list]
    for i_f, item in enumerate(file_list):

        raw_file_empty = check_file_empty(item.full)

        # raw file empty
        if raw_file_empty:
            empty_num += 1
            silent_remove(item.full)
            silent_remove(parsed_fn_list[i_f])
        # raw file not empty (unparsed or parsed)
        else:
            if os.path.exists(parsed_fn_list[i_f]):
                parsed_file_flg = not check_file_empty(parsed_fn_list[i_f]) and check_file_integrity(
                    parsed_fn_list[i_f]
                )
                # unparsed or corrupted
                if not parsed_file_flg:
                    unparsed_num += 1
                    silent_remove(parsed_fn_list[i_f])
                    check_unparsed_rawlist.append(item)
                # parsed
                else:
                    parsed_num += 1
            else:
                unparsed_num += 1
                check_unparsed_rawlist.append(item)
    total_num = parsed_num + unparsed_num + empty_num

    return parsed_num, unparsed_num, empty_num, total_num, check_unparsed_rawlist


def parse_func(task):
    threshold = 2000
    # file_writen_counter = 0
    corenlp_path = task.corenlp_path
    annotators = task.annotators
    port = task.port
    file_list = task.file_list
    parsed_root = task.parsed_root
    parsed_fn_list = [os.path.join(parsed_root, change_file_extension(f.fn)) for f in file_list]

    client, _ = get_corenlp_client(corenlp_path=corenlp_path, corenlp_port=port, annotators=annotators)

    for i_f, item in enumerate(file_list):
        if os.path.exists(parsed_fn_list[i_f]):
            if 'ner' not in anno:
                print('file:{} already parsed without ner, continue..'.format(parsed_fn_list[i_f]))
                continue
            no_link = True
            for i, line in enumerate(open(parsed_fn_list[i_f])):
                if i > 0:
                    if json.loads(line.strip()).get('mentions', None) is not None:
                        no_link = False
                    break
            if not no_link:
                print('file:{} already parsed and linked, continue..'.format(parsed_fn_list[i_f]))
                continue
        fn = item.full
        sentences = []
        para_lens = []
        for para in open(fn):
            sentences_unlinked = []
            for i_p in range(0, len(para), threshold):
                content = para[i_p:i_p + threshold].strip()
                if content is not None and len(content) > 0:
                    tmp = parse_sentense_with_stanford(content, client, annotators=anno)
                    sentences_unlinked.extend(tmp)

            if task.link_flg and not link_per_doc:
                link(sentences_unlinked)
            para_lens.append(len(sentences_unlinked))
            sentences.extend(sentences_unlinked)

        if task.link_flg and link_per_doc:
            link(sentences)

        if sentences == [] or para_lens == []:
            print('fn {} is empty'.format(fn))
            continue

        para_lens[0] += 1
        for i in range(1, len(para_lens)):
            para_lens[i] += para_lens[i - 1]

        if not os.path.exists(os.path.dirname(parsed_fn_list[i_f])):
            os.makedirs(os.path.dirname(parsed_fn_list[i_f]), exist_ok=True)

        with open(parsed_fn_list[i_f], 'w') as fw:
            fw.write(json.dumps({'sentence_lens': para_lens}) + '\n')
            for s in sentences:
                fw.write(json.dumps(s) + '\n')
                # file_writen_counter += 1
                # if file_writen_counter % 50 == 0:
                #     print(f'write {file_writen_counter} files already..')
    client.stop()


def main():
    parser = get_raw_process_parser()
    args = parser.parse_args()

    check_flg = args.check
    link_flg = args.link
    parse_flg = args.parse
    link_per_doc = args.link_per_doc

    print("process dataset:{}".format(args.data))
    if check_flg and not parse_flg:
        print('only check dataset')
    elif check_flg and parse_flg:
        print('check before parse dataset')
    elif not check_flg and parse_flg:
        print('only parse dataset')
    else:
        print('check and parse both not activated, error!')
        exit(-1)

    print(f'{"enable" if link_flg else "disable"} link')
    share_src = None
    if link_flg and parse_flg:
        print(f'link per {"doc" if link_per_doc else "paragraph"}')
        disam_fn = '/home/hkeaa/data/nel/basic_data/wiki_disambiguation_pages.txt'
        name_id_fn = '/home/hkeaa/data/nel/basic_data/wiki_name_id_map.txt'
        redirect_fn = '/home/hkeaa/data/nel/basic_data/wiki_redirects.txt'
        ment_ent_fn = '/home/data/corpora/wikipedia/ment_ent'
        person_fn = '/home/hkeaa/data/nel/basic_data/p_e_m_data/persons.txt'
        share_src = LinkSharedSource(disam_fn, redirect_fn, ment_ent_fn, person_fn)

    aser_root = '/home/data/corpora/aser/data'
    dataset_name = args.data
    raw_root = os.path.join(aser_root, dataset_name + '/raw')
    parsed_root = os.path.join(aser_root, dataset_name + '/parsed_new')
    worker_num = args.worker_num
    corenlp_path = '/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27'
    anno = ['tokenize', 'ssplit', 'pos', 'lemma', 'parse', 'ner']

    raw_inx_fn = os.path.join(raw_root, 'path_inx.json')
    if os.path.exists(raw_inx_fn) and not check_flg:
        file_name_list = load_paths(raw_inx_fn, raw_root)
    else:
        file_name_list = read_dir(raw_root)
        dump_paths(raw_inx_fn, file_name_list)
        print(f'saved in {raw_inx_fn}')

    print('all raw file number: {}'.format(len(file_name_list)))

    def chunk_list(l):
        chunk_size = min(20000, int(math.ceil(len(l) / (worker_num + 20.0))))
        for i in range(0, len(l), chunk_size):
            yield l[i:i + chunk_size]

    tasks = []
    server_num = 10

    if check_flg:
        print('check parsed result..')
        unparsed_fn = os.path.join(raw_root, 'path_inx.unparsed.json')
        parsed_num, unparsed_num, empty_num, total_num = 0, 0, 0, 0

        for i, fn_list in enumerate(chunk_list(file_name_list)):
            t = ParsingTask(fn_list, parsed_root, corenlp_path, 9101 + i % server_num, anno, link_flg)
            tasks.append(t)
        unparsed_list = []
        with Pool(worker_num) as pool:
            for res in tqdm(pool.imap_unordered(check_func, tasks), total=len(tasks)):
                res_parsed_num, res_unparsed_num, res_empty_num, res_total_num, res_unparsed_list = res
                parsed_num += res_parsed_num
                unparsed_num += res_unparsed_num
                empty_num += res_empty_num
                total_num += res_total_num
                unparsed_list.extend(res_unparsed_list)
        print('parsed num:{} prob:{:.4f}'.format(parsed_num, parsed_num / total_num))
        print('unparsed num:{} prob:{:.4f}'.format(unparsed_num, unparsed_num / total_num))
        print('empty num:{} prob:{:.4f}'.format(empty_num, empty_num / total_num))
        print('total num:{}'.format(total_num))
        dump_paths(unparsed_fn, unparsed_list)
        print(f'unparsed save in {unparsed_fn}')
        copyfile(unparsed_fn, raw_inx_fn)
        if parse_flg:
            file_name_list = unparsed_list
            print(f'read {len(file_name_list)} unparsed files after check')

    if parse_flg:
        div_num = args.chunk_size
        index = args.chunk_inx
        internal = len(file_name_list) // div_num + 1

        file_name_list = file_name_list[index * internal:(index + 1) * internal]

        print(f'parsing file num:{len(file_name_list)} from {index*internal} to {(index+1)*internal}')
        tasks = []

        for i, fn_list in enumerate(chunk_list(file_name_list)):
            t = ParsingTask(fn_list, parsed_root, corenlp_path, 9101 + i % server_num, anno, link_flg)
            tasks.append(t)

        # print('task num:{} file num:{} for {} workers'.format(len(tasks), file_num, worker_num))
        print(f'file {len(tasks[-1].file_list)} per task')
        import time
        t = time.time()
        with Pool(worker_num) as pool:
            res = pool.map_async(parse_func, tasks)
            res.get()
            res.wait()
        print(f'cost {time.time()-t}f')


if __name__ == '__main__':
    main()
