import re
import fermipy.defaults as defaults

def extract_typename(t):

    m = re.search('<(.*?) \'(.*?)\'>',str(t))
    if m is None:
        return ''
    else:
        return m.group(2)

for k, v in defaults.__dict__.items():

    if not isinstance(v,dict) or k.startswith('_'):
        continue

    f = open("config/%s.csv"%k, "w")

    if 'output' in k:
        for optname, optdict in v.items():
            if len(optdict) == 3:            
                f.write('``%s``\t`~%s`\t%s\n'%(optname, extract_typename(optdict[2]), optdict[1]))
            else:
                f.write('``%s``\t%s\t%s\n'%(optname,optdict[3],optdict[1]))
    else:
        for optname, optdict in sorted(v.items()):
            f.write('``%s``\t%s\t%s\n'%(optname,optdict[0],optdict[1]))

    f.close()
