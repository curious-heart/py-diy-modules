import pydicom as pd
import re
import os
import sys

_content_f_name = '00.contents.csv'
_cared_f_name_ptn = '^[0-9]+.dcm$'

_coding_type = 'gbk'
_key_str = 'key'
_desc_str = 'desc'
_col_keys_dict = \
[
    {_key_str: (0x0018, 0x1030), _desc_str: "体位（协议）"},
    {_key_str: (0x0018, 0x0060), _desc_str: "管电压(kv)"},
    {_key_str: (0x0018, 0x1151), _desc_str: "管电流(uA)"},
    {_key_str: (0x0018, 0x1150), _desc_str: "曝光时间(ms)"},
]

_usage_str = 'usage: {} folder'.format(__file__.split('\\')[-1])
if len(sys.argv) != 2:
    print(_usage_str)
    sys.exit(0)

_file_ptn_str = '文件'
c_tbl = [[d[_desc_str] for d in _col_keys_dict]]
c_tbl[0].insert(0, _file_ptn_str)

folder = sys.argv[1].replace('\\', '/')
f_tree = os.walk(folder)
for dp, dn, fn in f_tree:
    for f in fn:
        if re.match(_cared_f_name_ptn, f):
            file_ptn = dp.replace(folder, '')
            if '/' == file_ptn[0] or '\\' == file_ptn[0]: file_ptn = file_ptn[1:]
            file_ptn += '-' + f
            full_fn = os.path.join(dp, f)
            dcm_f = pd.dcmread(full_fn)
            this_line = [dcm_f.get_item(i[_key_str]).value.decode(_coding_type) for i in _col_keys_dict]
            this_line.insert(0, file_ptn)
            c_tbl.append(this_line)
result_fpn = folder + '/' + _content_f_name
result_f = open(result_fpn, "w")
for line in c_tbl:
    line_str = ""
    for item in line: line_str += str(item) + ','
    if ',' == line_str[-1]: line_str = line_str[:-1]
    print(line_str, file = result_f)
result_f.close()
