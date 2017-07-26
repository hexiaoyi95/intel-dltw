#!/bin/bash -e
# Colors
ESC_SEQ="\x1b["
COL_RESET=$ESC_SEQ"39;49;00m"
COL_RED=$ESC_SEQ"31;01m"
COL_GREEN=$ESC_SEQ"32;01m"
COL_YELLOW=$ESC_SEQ"33;01m"
COL_BLUE=$ESC_SEQ"34;01m"
COL_MAGENTA=$ESC_SEQ"35;01m"
COL_CYAN=$ESC_SEQ"36;01m""]"

echo -e "${COL_BLUE}Installing python dependency...${COL_RESET}"
if pip --help | grep "trusted-host" ;then
    pip install --user --proxy http://child-prc.intel.com:913  --trusted-host pypi.douban.com -i http://pypi.douban.com/simple -r requirements.txt
else
    pip install --user --proxy http://child-prc.intel.com:913  -i http://pypi.douban.com/simple -r requirements.txt
fi

echo -e "${COL_BLUE}Downloading data ...${COL_RESET}"
http_proxy="" wget -O dl-data.tar http://test132.sh.intel.com:666/dltw-data.tar
tar -xf dl-data.tar
rm dl-data.tar

echo -e "${COL_BLUE}Downloading models ...${COL_RESET}"
http_proxy="" wget -O dl-models.tar http://test132.sh.intel.com:666/dltw-models.tar
tar -xf dl-models.tar
rm dl-models.tar

mkdir -p dl-frameworks

echo -e "${COL_GREEN}Finished.${COL_RESET}"
