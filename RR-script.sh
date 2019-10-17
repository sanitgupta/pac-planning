#! /bin/bash

counter=1
while [ $counter -le 100 ]
do
	python main.py $1 rr
	((counter++))
	echo counter
done

echo All done