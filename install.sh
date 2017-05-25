#!/bin/bash -x
print "Installing python dependency..."
pip install --user --proxy http://child-prc.intel.com:913  --trusted-host pypi.douban.com -i http://pypi.douban.com/simple -r requirements.txt

print "Downloading data ..."
wget -O dl-data.tar http://test132.sh.intel.com:666/dltw-data.tar
tar -xf dl-data.tar
rm dl-data.tar

print "Downloading models ..."
wget -O dl-models.tar http://test132.sh.intel.com:666/dltw-models.tar
tar -xf dl-models.tar
rm dl-models.tar

mkdir dl-frameworks
