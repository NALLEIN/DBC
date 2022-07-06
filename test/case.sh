for file in `find /media/jianghao/Elements/datasets/DIV2K/DIV2K_valid_HR | grep ".*png"`
do
    echo $(basename $file) >> /home/jianghao/Code/bytedance/DBC/codes/data/meta_info/DIV2K_validHR.txt
done