# download uci 121 datasets
mkdir data/uci
cd data/uci && wget -nc http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz
tar xkf data.tar.gz 2>&1 | head -n 5 && echo ...

