#!/bin/bash
# for quality in {1..8..1}
# do
#     python3 codes/test.py -opt="codes/options/test/test_resize_q$quality_factorized.yml" >> test/facorizedhyperprior.txt
# done 
# for quality in {1..8..1}
# do
#     python3 codes/test.py -opt="codes/options/test/test_resize_q$quality.yml" >> test/scalehyperprior.txt
# done 
python3 test/BD-rate.py -savepath="test/" -path1="facorizedhyperprior.txt" -path2="scalehyperprior.txt"