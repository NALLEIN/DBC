# #for file in `find /media/20210822/good_cases | grep ".*391477222_S000.mp4"`
# for file in `find /home/jianghao/Dataset/4k_video | grep ".*mp4"`
# do
# 	echo $(basename $file .mp4)
# 	tmppath="/home/jianghao/Code/Graduation/540p/videos/"
# 	echo $tmppath$(basename $file .mp4)
# 	mkdir $tmppath$(basename $file .mp4)
# 	# /home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -i $file -pix_fmt yuv420p -frames 100 decode.yuv -y # > /dev/null 2>&1

# 	# /home/jianghao/anaconda3/envs/pytorch/bin/python /home/jianghao/Code/Graduation/540p/codes/test_forward_yuv.py \
# 	# -opt /home/jianghao/Code/Graduation/540p/codes/options/test/test_EDSR_forward.yml #  > /dev/null 2>&1
# 	for qp in {22..27..1}
# 	do
# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 1920x1080 -pix_fmt yuv420p -i decode_net.yuv \
# 		-c:v libx265 -x265-params "qp=${qp}" codec_net.hevc -y > $tmppath$(basename $file .mp4)/transcode_net_${qp} 2>&1

# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -i codec_net.hevc codec_net.yuv -y > /dev/null 2>&1

# 		/home/jianghao/anaconda3/envs/pytorch/bin/python /home/jianghao/Code/Graduation/540p/codes/scale_transfer.py \
# 		-video_path /home/jianghao/Code/Graduation/540p/test/codec_net.yuv  -mode up > /dev/null 2>&1
		
# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 3840x2160 -pix_fmt yuv420p -i decode.yuv -s 3840x2160 -pix_fmt yuv420p  -i codec_net_up.yuv \
# 		-filter_complex "[1:v]scale=1920:1080[scale];[scale][0:v]psnr=stats_file="$tmppath$(basename $file .mp4)/psnr_net_${qp} \
# 		-f null /dev/null > $tmppath$(basename $file .mp4)/Avg_psnr_net_${qp} 2>&1

# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 3840x2160 -pix_fmt yuv420p -i decode.yuv -s 3840x2160 -pix_fmt yuv420p  -i codec_net_up.yuv \
# 		-filter_complex "[1:v]scale=1920:1080[scale];[scale][0:v]ssim=stats_file="$tmppath$(basename $file .mp4)/ssim_net_${qp} \
# 		-f null /dev/null > $tmppath$(basename $file .mp4)/Avg_ssim_net_${qp} 2>&1

# 		/home/jianghao/anaconda3/envs/pytorch/bin/python /home/jianghao/Code/Graduation/540p/codes/scale_transfer.py \
# 		-video_path /home/jianghao/Code/Graduation/540p/test/decode.yuv  -mode down > /dev/null 2>&1

# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 1920x1080 -pix_fmt yuv420p -i decode_down.yuv \
# 		-c:v libx265 -x265-params "qp=${qp}" codec.hevc -y > $tmppath$(basename $file .mp4)/transcode_lanczos_${qp} 2>&1

# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -i codec.hevc codec.yuv -y > /dev/null 2>&1

# 		/home/jianghao/anaconda3/envs/pytorch/bin/python /home/jianghao/Code/Graduation/540p/codes/scale_transfer.py \
# 		-video_path /home/jianghao/Code/Graduation/540p/test/codec.yuv  -mode up > /dev/null 2>&1

# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 3840x2160 -pix_fmt yuv420p -i decode.yuv -s 3840x2160 -pix_fmt yuv420p  -i codec_up.yuv \
# 		-filter_complex "[1:v]scale=1920:1080[scale];[scale][0:v]psnr=stats_file="$tmppath$(basename $file .mp4)/psnr_lanczos_${qp} \
# 		-f null /dev/null > $tmppath$(basename $file .mp4)/Avg_psnr_lanczos_${qp} 2>&1

# 		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 3840x2160 -pix_fmt yuv420p -i decode.yuv -s 3840x2160 -pix_fmt yuv420p  -i codec_up.yuv \
# 		-filter_complex "[1:v]scale=1920:1080[scale];[scale][0:v]ssim=stats_file="$tmppath$(basename $file .mp4)/ssim_lanczos_${qp} \
# 		-f null /dev/null > $tmppath$(basename $file .mp4)/Avg_ssim_lanczos_${qp} 2>&1
# 	done
# 	# python BD-rate.py -path $tmppath$(basename $file .mp4) >> TempTest_mp4.log
# done

# for file in `find /media/jianghao/MyPassport/xiph_422 | grep ".*yuv"`
# do
#     dirpath="/media/jianghao/MyPassport/xiph_yuv"
#     outfile=$dirpath/$(basename $file)
#     echo $dirpath/$(basename $file)
#     /home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 1920x1080 -pix_fmt yuv422p -i $file -s 1920x1080 -pix_fmt yuv420p $outfile -y
# done

# for file in `cat xiph.txt`
# do
#     dirpath="/media/jianghao/MyPassport/xiph_422"
#     echo $file
#     `mv $file $dirpath/$(basename $file)`
# done

for file in `find /media/jianghao/Samsung_T5/bilibili_test | grep ".*origvideo"`
do
    echo $file >> test_videos.txt
done