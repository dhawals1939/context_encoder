error(){
	echo 'invalid number of args passed'
	echo 'arg-1 train_data_path'
	echo 'arg-2 result_path'
	echo 'arg-3 log_path'
	echo 'arg-4 check_point_path'
	echo 'arg-5 log epoch count'
}
if [ $# -ne 5 ]
then
	error
else
	python3 context_encoder.py $1 $2 $3 $4 $5
	if [ $? -ne 0 ]
	then
		error
	fi
fi
