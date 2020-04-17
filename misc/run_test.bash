if [ $# -ne 3 ]
then
	echo 'invalid number of args passed'
	echo 'arg-1 model_path generator in pt'
	echo 'arg-2 test_path dir'
	echo 'arg-3 test_output_path dir'
else
	python3 test.py $1 $2 $3
fi
