for file in `find /home/jianghao/Dataset/Flickr2K_HR | grep ".*png"`
do
    echo $(basename $file) >> /DATA/jianghao/Code/DBC/codes/data/meta_info/flicker2k.txt
done