
#export ST_URL="https://www.dropbox.com/s/b5z0wfc0xt9n0ha/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz"
#export ST_URL="https://www.slac.stanford.edu/~mdwood/fermipy/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz"

conda install -c conda-forge/label/cf201901 -c fermi fermitools

#if [[ ! -d "$HOME/ScienceTools" || -n "$1" ]]; then
#if [ -n "$1" ]; then
#if (true); then
#    wget -nc $ST_URL -O $HOME/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz
#    cd $HOME; tar xzf $HOME/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA-light.tar.gz
#    mv $HOME/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-root $HOME/ScienceTools
#fi
