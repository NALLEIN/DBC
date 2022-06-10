for file in `find /media/jianghao/Samsung_T5/dataset/clic/all | grep ".*png"`
do
    echo $(basename $file) >> /home/jianghao/Code/bytedance/DBC/codes/data/meta_info/clic.txt
done