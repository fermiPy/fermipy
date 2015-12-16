
#export ST_URL="https://www.dropbox.com/s/b5z0wfc0xt9n0ha/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz"
export ST_URL="https://www.slac.stanford.edu/~mdwood/fermipy/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz"

if [ ! -d "$HOME/ScienceTools" ]; then
    wget -nc $ST_URL -O $HOME/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz
    cd $HOME; tar xzf $HOME/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz
    mv $HOME/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-root $HOME/ScienceTools
fi
