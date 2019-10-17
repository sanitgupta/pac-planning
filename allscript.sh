#! /bin/bash

echo "Executing mdp-01 lucb"
time python MDPclass.py mdp-01.txt lucb 3 > observations/mdp-01-lucb.txt

echo "Executing mdp-01 rr"
time python MDPclass.py mdp-01.txt rr 3 > observations/mdp-01-rr.txt

echo "Executing mdp-01 fiechter"
time python MDPclass.py mdp-01.txt fiechter 3 > observations/mdp-01-fiechter.txt

echo "Executing mdp-SixArms lucb"
time python MDPclass.py mdp-SixArms.txt lucb 3 > observations/mdp-SixArms-lucb.txt

echo "Executing mdp-SixArms rr"
time python MDPclass.py mdp-SixArms.txt rr 3 > observations/mdp-SixArms-rr.txt

echo "Executing mdp-SixArms fiechter"
time python MDPclass.py mdp-SixArms.txt fiechter 3 > observations/mdp-SixArms-fiechter.txt

echo "Executing mdp-CasinoLand lucb"
time python MDPclass.py mdp-CasinoLand.txt lucb 3 > observations/mdp-CasinoLand-lucb.txt

echo "Executing mdp-CasinoLand rr"
time python MDPclass.py mdp-CasinoLand.txt rr 3 > observations/mdp-CasinoLand-rr.txt

echo "Executing mdp-CasinoLand fiechter"
time python MDPclass.py mdp-CasinoLand.txt fiechter 3 > observations/mdp-CasinoLand-fiechter.txt


python plot.py 4 riverswim-rr.txt riverswim-lucbeps.txt riverswim-fiechter.txt riverswim-ddv.txt

python plot.py 6 SixArms-rr.txt SixArms-lucb.txt SixArms-lucbeps.txt SixArms-fiechter.txt SixArms-mbie.txt SixArms-ddv.txt

python plot.py 5 CasinoLand-rr.txt CasinoLand-lucbeps.txt CasinoLand-fiechter.txt CasinoLand-mbie.txt CasinoLand-ddv.txt
