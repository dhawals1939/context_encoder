if [ $# -ne 7 ]
then
	echo 'invalid number of args passed'
	echo 'arg-1 train_data_path'
	echo 'arg-2 result_path'
	echo 'arg-3 log_path'
	echo 'arg-4 check_point_path'
	echo 'arg-5 final_model_save_path'
	echo 'arg-6 test_data_path'
	echo 'arg-7 test_output_path'
else
	python3 context_encoder.py $1 $2 $3 $4 $5 $6 $7
fi
