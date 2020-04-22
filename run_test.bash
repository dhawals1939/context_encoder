error(){
	echo 'invalid number of args passed or argument are invalid'
	echo 'arg-1 check point path'
	echo 'arg-2 test__dir'
	echo 'arg-3 test_results_dir'
}
if [ $# -ne 3 ]
then
	error
else
	python3 context_encoder_test.py $1 $2 $3
	if [ $? -ne 0 ]
	then
		error
	fi
fi
