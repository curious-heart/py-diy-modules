import os
import sys

def s_name(fp, s_str, d_str, recur = True):
    if recur:
        triples = os.walk(fp)
    else:
        triples = [list(os.walk(fp))[0]]
    for dp, dn, fn in triples: 
        for f in fn:
            new_f = f
            new_f = new_f.replace(s_str, d_str)
            if new_f != f :
                old_fpn = os.path.join(dp,f)
                new_fpn = os.path.join(dp, new_f)
                os.rename(old_fpn, new_fpn)
                print(old_fpn + "    -->    " + new_fpn)

help_str = "usage:" + \
            "rn" + " 路径 源字符串 目标字符串 [-c]"
if len(sys.argv) < 4:
    print(help_str)
    sys.exit(0)
recur = True
if len(sys.argv) == 5:
    if "-c" != sys.argv[4]:
        print(help_str)
        sys.exit(0)
    else:
        recur = False
fpn, s_str, d_str = sys.argv[1], sys.argv[2], sys.argv[3]
s_name(fpn, s_str, d_str, recur)
