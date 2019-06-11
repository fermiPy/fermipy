 
export ST_URL="https://www.dropbox.com/s/b5z0wfc0xt9n0ha/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz"
export ST_PACKED="ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz"
export ST_UNPACK="ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-root"

#export ST_URL="https://www.slac.stanford.edu/~echarles/ScienceTools-v11r5p3-fssc-20180124-x86_64-unknown-linux-gnu-libc2.17_lite.tgz"
#export ST_PACKED="ScienceTools-v11r5p3-fssc-20180124-x86_64-unknown-linux-gnu-libc2.17_lite.tgz"
#export ST_UNPACK="ScienceTools-v11r5p3-fssc-20180124-x86_64-unknown-linux-gnu-libc2.17"


if [[ -n $1 ]]; then
    ST_PATH=$1
else
    ST_PATH=$HOME
fi

if [[ ! -d "$ST_PATH/ScienceTools" ]]; then
    wget -nc $ST_URL -O $ST_PATH/$ST_PACKED
    cd $ST_PATH; tar xzf $ST_PATH/$ST_PACKED
    mv $ST_PATH/$ST_UNPACK $ST_PATH/ScienceTools
fi
