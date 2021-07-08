# download uci 121 datasets
# command: sh data/download_uci_datasets.sh

mkdir data/uci
cd data/uci && wget -nc http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz
tar xkf data.tar.gz 2>&1 | head -n 5 && echo ...

