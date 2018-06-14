#/bin/bash
set -x
LINK=https://download.pytorch.org/tutorial/hymenoptera_data.zip
wget -c $LINK;
unzip hymenoptera_data.zip;
rm hymenoptera_data.zip
