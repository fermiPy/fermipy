import fermipy.defaults as defaults

for k, v in defaults.__dict__.items():

    if not isinstance(v,dict) or k.startswith('_'):
        continue

    f = open("config/%s.csv"%k, "w")

    if 'output' in k:
        for optname, optdict in v.items():
            f.write('``%s``\t%s\t%s\n'%(optname,optdict[3],optdict[1]))
    else:
        for optname, optdict in sorted(v.items()):
            f.write('``%s``\t%s\t%s\n'%(optname,optdict[0],optdict[1]))

    f.close()
